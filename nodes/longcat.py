"""LongCat voice cloning, segmentation and loudness matching."""
from __future__ import annotations

import logging
import math
import re
from typing import Any, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# —— 依赖我们自己的离线加载器 ——
from .offline_loader import (
    approx_duration_from_text,
    is_audiodit_enabled,
    load_local_model,
    model_names_or_default,
    normalize_text,
    numpy_audio_to_comfy,
    register_model_folder,
    resolve_device,
    tokenizer_names_or_default,
)

# —— 复用移植到本插件内部的 model_cache 工具（缓存、卸载、动态显存） ——
from .model_cache import (
    cancel_event,
    get_cache_key,
    get_cached_model,
    is_offloaded,
    offload_model_to_cpu,
    resume_model_to_cuda,
    set_cached_model,
    set_generating,
    set_unload_mode,
    unload_model,
)
from .ui_labels import chinese_inputs, internal_choice, validate_choices


logger = logging.getLogger("LongCatHuan")

# —— 仅当明确启用时才注册 folder_paths 目录（与原插件共享） ——
# 这样不使用 TTS 的用户完全不会触发 folder_paths / models 扫描
if is_audiodit_enabled():
    register_model_folder()


# ---------------- Comfy 环境 / 进度条 ----------------
try:
    from comfy.utils import ProgressBar  # type: ignore
    _PBAR = True
except Exception:
    _PBAR = False

try:
    import comfy.model_management as mm  # type: ignore
    _MM = True
except Exception:
    _MM = False


# 模型卸载模式三选一：
#   full_unload: 完全卸载（del + empty_cache），显存彻底释放，下次推理重新加载
#   offload_cpu: offload 到 CPU（保留权重），显存释放但模型仍在内存，下次推理 resume 回 GPU
#   keep_gpu:    完全保持 GPU，永不离开，占显存换最快连续推理速度
UNLOAD_MODES = ["full_unload", "offload_cpu", "keep_gpu"]
UNLOAD_MODE_DEFAULT = "offload_cpu"
UNLOAD_MODE_TOOLTIP = (
    "模型卸载模式：\n"
    "  full_unload: 完全卸载（彻底释放显存，下次推理重新加载，最慢但显存最干净）\n"
    "  offload_cpu: offload 到 CPU（释放显存但保留权重，下次推理快速 resume，平衡）\n"
    "  keep_gpu:    完全保持 GPU（永不离开 GPU，占显存换最快连续推理速度）"
)


# ---------------- 共享辅助函数 ----------------
def comfy_audio_to_tensor(audio_dict: dict, target_sr: int) -> torch.Tensor:
    waveform = audio_dict["waveform"]
    source_sr = audio_dict["sample_rate"]
    wav = waveform[0].float()
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    elif wav.shape[0] == 1:
        pass
    else:
        wav = wav.unsqueeze(0)
    wav_np = wav.squeeze(0).numpy()
    if source_sr != target_sr:
        import librosa  # type: ignore
        wav_np = librosa.resample(wav_np, orig_sr=source_sr, target_sr=target_sr)
    return torch.from_numpy(wav_np).unsqueeze(0)


# 扫描委托到 offline_loader.model_names_or_default（与 tokenizer 共享 verbose 机制）。
# 保持本文件内代码继续使用 _model_names_or_default 名字，避免改动面太大。
def _model_names_or_default(verbose: bool = False) -> list[str]:
    return model_names_or_default(verbose=verbose)


def _interrupt_check():
    if _MM:
        try:
            mm.throw_exception_if_processing_interrupted()
        except Exception:
            cancel_event.set()
            raise


# ---------------- 长文本分句工具 ----------------
def _split_text_into_segments(
    text: str,
    available_duration: float,
    sr: int,
    full_hop: int,
    duration_multiplier: float = 1.0,
) -> list[tuple[str, int]]:
    """
    将长文本按标点切分为多个小段，每段时长不超过 available_duration。
    返回 [(segment_text, duration_in_latent_frames), ...]

    策略：
    1. 按句末标点（。！？.!?；;）切分
    2. 单句超长则按逗号/冒号切分
    3. 仍超长则按字数硬切
    4. 将短句合批，尽量填满 available_duration

    duration_multiplier: 时长放大系数（用于音色克隆时匹配参考音频语速）。
        每段的 dur_frames 会乘以此系数，让模型生成更长的音频（语速更慢）。
    """
    # 安全余量系数：所有分段只用 85% 可用时长的字数，留 15% 防止吞字
    SAFE_FACTOR = 0.85
    safe_duration = available_duration * SAFE_FACTOR

    # Step 1: 按句末标点切分
    raw_sentences = re.split(r'(?<=[。！？.!?；;])\s*', text)
    raw_sentences = [s.strip() for s in raw_sentences if s.strip()]

    # Step 2: 检查每句时长，超长则继续切分
    sentences: list[str] = []
    for sent in raw_sentences:
        sent_dur = approx_duration_from_text(sent, max_duration=math.inf)
        if sent_dur <= safe_duration:
            sentences.append(sent)
        else:
            # 按逗号/冒号切分
            comma_parts = re.split(r'(?<=[，,、：:])\s*', sent)
            comma_parts = [p.strip() for p in comma_parts if p.strip()]
            for cp in comma_parts:
                cp_dur = approx_duration_from_text(cp, max_duration=math.inf)
                if cp_dur <= safe_duration:
                    sentences.append(cp)
                else:
                    # 仍超长 → 按字数硬切（按比例估算每段字数）
                    max_chars = max(1, int(safe_duration / 0.21))
                    for i in range(0, len(cp), max_chars):
                        chunk = cp[i : i + max_chars]
                        if chunk:
                            sentences.append(chunk)

    # Step 3: 将短句合批为不超时的段（使用 safe_duration 而非 available_duration）
    segments: list[tuple[str, int]] = []
    cur_text = ""
    cur_dur = 0.0

    for sent in sentences:
        sent_dur = approx_duration_from_text(sent, max_duration=math.inf)
        if cur_dur + sent_dur > safe_duration and cur_text:
            dur_frames = max(1, math.ceil(cur_dur * sr / full_hop * duration_multiplier))
            segments.append((cur_text, dur_frames))
            cur_text = sent
            cur_dur = sent_dur
        else:
            cur_text = (cur_text + sent) if cur_text else sent
            cur_dur += sent_dur

    if cur_text:
        dur_frames = max(1, math.ceil(cur_dur * sr / full_hop * duration_multiplier))
        segments.append((cur_text, dur_frames))

    return segments


def _concat_audio_segments(chunks: list[np.ndarray], pause: float, sr: int) -> np.ndarray:
    """将多段音频拼接，段间插入 pause 秒静音。"""
    if not chunks:
        return np.zeros(int(sr * 0.1), dtype=np.float32)
    if pause <= 0:
        return np.concatenate(chunks, axis=-1)
    sil = np.zeros(int(sr * pause), dtype=np.float32)
    out_parts: list[np.ndarray] = []
    for i, c in enumerate(chunks):
        out_parts.append(c)
        if i < len(chunks) - 1:
            out_parts.append(sil)
    return np.concatenate(out_parts, axis=-1)


# ---------------- 响度保护机制（输入输出音量相等，与 indextts 同类保护） ----------------
# 零样本（无参考音频）时的固定归一化目标：-20 dBFS ≈ 0.1
_LOUDNESS_TARGET_RMS = 10 ** (-20 / 20)
# 单次增益钳制 ±12dB：防止把生成底噪放大成爆破音，或过度压低输出
_LOUDNESS_MAX_GAIN_DB = 12.0
# 防削波峰值上限
_LOUDNESS_PEAK_LIMIT = 0.98


def _compute_rms(wav) -> float:
    """计算音频 RMS 响度。支持 np.ndarray / torch.Tensor；空/静音返回 0。"""
    if isinstance(wav, torch.Tensor):
        w = wav.detach().cpu().float().numpy().astype(np.float64)
    else:
        w = np.asarray(wav, dtype=np.float64)
    if w.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(w * w)))


def _match_loudness(wav: np.ndarray, target_rms: float) -> np.ndarray:
    """响度保护：把生成音频 RMS 对齐到 target_rms（输入输出音量相等）。

    - 增益钳制 ±12dB（防爆音/过度压低）
    - 防削波：增益后峰值 > 0.98 时按峰值自动回退
    - target_rms 无效或自身为静音时原样返回
    """
    if wav is None or wav.size == 0 or target_rms <= 0.0:
        return wav
    cur = _compute_rms(wav)
    if cur < 1e-8:
        return wav
    gain = float(np.clip(
        target_rms / cur,
        10 ** (-_LOUDNESS_MAX_GAIN_DB / 20),
        10 ** (_LOUDNESS_MAX_GAIN_DB / 20),
    ))
    out = wav.astype(np.float32) * gain
    peak = float(np.max(np.abs(out)))
    if peak > _LOUDNESS_PEAK_LIMIT:
        out = out * (_LOUDNESS_PEAK_LIMIT / peak)
    return out


# ====================================================================
# 1. 零样本 TTS（离线版）
# ====================================================================

def _check_model_selection( model_path: str, tokenizer: str) -> None:
    """执行节点入口处才做的检查（启动阶段不做，避免刷屏）：
    1) 明确设置 LONGCAT_HUAN_DISABLE=1 -> 错误提示用户已主动关闭
    2) 模型/tokenizer 仍为占位项 -> verbose=True 触发一次性 info + 清晰报错
    """
    if not is_audiodit_enabled():
        raise RuntimeError(
            "[LongCat-Huan] 已通过环境变量 LONGCAT_HUAN_DISABLE=1 关闭。\n"
            "如需使用 LongCat-AudioDiT TTS，请先取消该变量（或置为 0/off）后重启 ComfyUI。"
        )
    # 执行时 verbose=True：若缺失，这里才会第一次（且仅一次）打印 info 级提示
    _model_names_or_default(verbose=True)
    tokenizer_names_or_default(verbose=True)
    if model_path.startswith("（请先放入"):
        raise FileNotFoundError(
            "[LongCat-Huan] 未选择任何可用模型。\n"
            "请把 LongCat-AudioDiT 模型目录放到 ComfyUI/models/audiodit/ 后刷新节点列表。"
        )
    if tokenizer.startswith("（请先放入"):
        raise FileNotFoundError(
            "[LongCat-Huan] 未选择任何可用 tokenizer。\n"
            "请把 UMT5 tokenizer 放到 ComfyUI/models/audiodit/umt5-base-tokenizer/ 后刷新节点列表。"
        )


class LongCatVoiceCloneHuan:
    """LongCat-Huan 音色克隆 TTS（严格离线，自动分句支持长文本）。参考音频 3–15 秒效果最佳。"""

    @classmethod
    def INPUT_TYPES(cls):
        model_names = _model_names_or_default()
        tokenizer_names = tokenizer_names_or_default()
        return chinese_inputs({
            "required": {
                "model_path": (
                    model_names,
                    {"tooltip": (
                        "【严格离线】仅列出 ComfyUI/models/audiodit/ 下的本地模型。"
                    )},
                ),
                "tokenizer": (
                    tokenizer_names,
                    {
                        "tooltip": (
                            "文本分词器目录（UMT5 tokenizer）。仅列出本地已存在的目录。"
                        ),
                    },
                ),
                "text": (
                    "STRING", {
                        "multiline": True,
                        "default": "这是用参考音色克隆出来的语音，可以自由改变要说的文本内容。",
                        "tooltip": "目标文本（用克隆出来的音色朗读）。自动分句，支持长文本。",
                    },
                ),
                "prompt_audio": (
                    "AUDIO", {"tooltip": "参考音频（要克隆的音色）。3–15 秒效果最佳。"},
                ),
                "prompt_text": (
                    "STRING", {
                        "multiline": True,
                        "default": "",
                        "tooltip": "参考音频的文字转录（强烈建议提供，会显著提升克隆质量）。",
                    },
                ),
                "steps": ("INT", {"default": 16, "min": 4, "max": 64, "step": 1,
                                 "tooltip": "ODE Euler 步数。"}),
                "guidance_strength": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 10.0, "step": 0.5,
                                                "tooltip": "CFG/APG 引导强度。"}),
                "guidance_method": (["cfg", "apg"], {"default": "apg",
                                                     "tooltip": "音色克隆推荐 apg。"}),
                "device": (["auto", "cuda", "cpu", "mps"], {"default": "auto"}),
                "dtype": (["auto", "bf16", "fp16", "fp32"], {"default": "auto",
                                                             "tooltip": "音色克隆推荐 bf16，fp16 会数值溢出。"}),
                "attention": (["auto", "sdpa", "sage_attention", "flash_attention"], {"default": "auto"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
                "unload_mode": (
                    UNLOAD_MODES,
                    {"default": UNLOAD_MODE_DEFAULT, "tooltip": UNLOAD_MODE_TOOLTIP},
                ),
                "pause_between_segments": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 2.0, "step": 0.1,
                    "tooltip": "自动分句时各段之间的静音秒数。长文本分句拼接时使用。"}),
            },
            "optional": {
                "volume_protect": ("BOOLEAN", {"default": True,
                    "tooltip": "响度保护：输出 RMS 对齐参考音频 RMS（输入输出音量相等），\n同时长文本各段天然对齐到同一目标，段间音量一致。\n含 ±12dB 增益钳制与防削波保护。"}),
            },
        })

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("合成音频",)
    FUNCTION = "generate"
    CATEGORY = "LongCat-Huan"

    @classmethod
    def VALIDATE_INPUTS(cls, guidance_method, device, dtype, attention, unload_mode):
        return validate_choices({
            "guidance_method": guidance_method, "device": device, "dtype": dtype,
            "attention": attention, "unload_mode": unload_mode,
        })

    def generate(
        self,
        model_path: str,
        tokenizer: str,
        text: str,
        prompt_audio: dict,
        prompt_text: str,
        steps: int,
        guidance_strength: float,
        guidance_method: str,
        device: str,
        dtype: str,
        attention: str,
        seed: int,
        unload_mode: str,
        pause_between_segments: float = 0.3,
        volume_protect: bool = True,
    ) -> Tuple[dict]:
        guidance_method = internal_choice("guidance_method", guidance_method)
        device = internal_choice("device", device)
        dtype = internal_choice("dtype", dtype)
        attention = internal_choice("attention", attention)
        unload_mode = internal_choice("unload_mode", unload_mode)
        cancel_event.clear()
        _interrupt_check()
        # 启动阶段不打印缺失警告，执行时才检查启用状态并一次性提示
        _check_model_selection(model_path, tokenizer)

        if not text.strip():
            raise ValueError("[LongCat-Huan] 目标文本不能为空。")
        if not prompt_text.strip():
            logger.warning(
                "未提供参考音频转录（prompt_text）。克隆质量可能下降，强烈建议填写。"
            )

        if dtype == "fp16":
            logger.warning(
                "音色克隆不支持 fp16（latent 条件路径会数值溢出），自动升级为 bf16。"
            )
            dtype = "bf16"

        set_generating(True)
        try:
            model, tokenizer = self._get_model(
                model_path, tokenizer, device, dtype, attention, unload_mode
            )

            sr = model.config.sampling_rate
            full_hop = model.config.latent_hop
            max_duration = model.config.max_wav_duration

            logger.info("编码参考音频…")
            prompt_wav = comfy_audio_to_tensor(prompt_audio, sr).to(model.device)
            # 响度保护：以参考音频 RMS 为目标（输入输出音量相等）
            ref_rms = _compute_rms(prompt_wav) if volume_protect else 0.0
            if volume_protect:
                logger.info(f"响度保护已开启：输出将匹配参考音频 RMS={ref_rms:.4f}")
            prompt_duration = prompt_wav.shape[-1] / sr
            if prompt_duration > 30:
                logger.warning(
                    f"参考音频时长 {prompt_duration:.1f}s > 30s。可能影响生成，建议裁剪。"
                )

            # 预编码参考音频 VAE latent（所有段共享，提前到分段之前以便计算 ratio）
            prompt_audio_tensor = prompt_wav.unsqueeze(0)
            off = 3
            pw = prompt_wav.clone()
            if pw.shape[-1] % full_hop != 0:
                pw = F.pad(pw, (0, full_hop - pw.shape[-1] % full_hop))
            pw = F.pad(pw, (0, full_hop * off))
            with torch.no_grad():
                plt = model.vae.encode(pw.unsqueeze(0))
            if off:
                plt = plt[..., :-off]
            prompt_dur = plt.shape[-1]
            prompt_time = prompt_dur * full_hop / sr

            # ratio: 参考音频实际语速 vs 估算语速，用于放大目标文本时长（匹配原 LongCat 逻辑）
            prompt_text_norm = normalize_text(prompt_text) if prompt_text.strip() else ""
            ratio = 1.0
            if prompt_text_norm:
                approx_pd = approx_duration_from_text(prompt_text_norm, max_duration=max_duration)
                ratio = float(np.clip(prompt_time / max(approx_pd, 0.1), 1.0, 1.5))
                logger.info(
                    f"参考音频语速系数 ratio={ratio:.2f}"
                    f"（prompt_time={prompt_time:.2f}s, approx_pd={approx_pd:.2f}s）"
                )

            available_duration = max_duration - prompt_time
            if available_duration <= 1.0:
                raise ValueError(
                    f"可用时长 {available_duration:.1f}s 太短（参考音频占 {prompt_time:.1f}s）。"
                    f"请缩短参考音频或增加文本。"
                )

            text_norm = normalize_text(text)
            logger.info(f"LongCat 音色克隆: {text_norm[:80]}{'…' if len(text_norm) > 80 else ''}")

            # 自动分句（ratio 放大每段 dur_frames，让语速匹配参考音频）
            # 关键：分段阈值按 ratio 反向缩小，使每段 cur_dur * ratio 不超过 available_duration，
            # 避免 duration = seg_dur_frames + prompt_dur 撞 max_dur_frames 上限被截断吞字。
            # 数学保证：cur_dur ≤ (available_duration / ratio) * 0.85
            #         → cur_dur * ratio ≤ available_duration * 0.85 < available_duration
            seg_threshold = available_duration / ratio if ratio > 0 else available_duration
            segments = _split_text_into_segments(
                text_norm, seg_threshold, sr, full_hop, duration_multiplier=ratio
            )
            if len(segments) > 1:
                logger.info(
                    f"文本自动分为 {len(segments)} 段（每段 ≤ {seg_threshold:.1f}s, ratio={ratio:.2f}）"
                )
            else:
                logger.info(f"单段生成，预计 ≤ {seg_threshold:.1f}s, ratio={ratio:.2f}")

            actual_seed = seed if seed != 0 else torch.randint(0, 2**31, (1,)).item()
            torch.manual_seed(actual_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(actual_seed)


            # 逐段生成
            chunks: list[np.ndarray] = []
            total_segments = len(segments)
            pbar = ProgressBar(total_segments + 1) if _PBAR else None

            for seg_idx, (seg_text, seg_dur_frames) in enumerate(segments):
                _interrupt_check()
                seg_dur_sec = seg_dur_frames * full_hop / sr
                logger.info(f"段 {seg_idx + 1}/{total_segments}: '{seg_text[:40]}…' 约 {seg_dur_sec:.1f}s")

                seg_full_text = f"{prompt_text_norm} {seg_text}" if prompt_text_norm else seg_text
                inputs = tokenizer([seg_full_text], padding="longest", return_tensors="pt")
                input_ids = inputs.input_ids.to(model.device)
                attention_mask = inputs.attention_mask.to(model.device)

                duration = seg_dur_frames + prompt_dur
                max_dur_frames = int(max_duration * sr // full_hop)
                if duration > max_dur_frames:
                    logger.warning(
                        "[LongCat-Huan] 段 %d/%d 时长 %d 帧超过最大限制 %d 帧（已截断，可能吞字）：'%s'",
                        seg_idx + 1, total_segments, duration, max_dur_frames, seg_text[:40],
                    )
                duration = min(duration, max_dur_frames)

                seg_seed = (actual_seed + seg_idx * 1000003) % (2**31)
                with torch.no_grad():
                    output = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        prompt_audio=prompt_audio_tensor,
                        duration=duration,
                        steps=steps,
                        cfg_strength=guidance_strength,
                        guidance_method=guidance_method,
                        seed=seg_seed,
                    )
                wav = output.waveform.squeeze().detach().cpu().numpy()
                # 响度保护：输出对齐参考音频 RMS（模型输出已剥离参考前缀，纯生成音频）
                if volume_protect and ref_rms > 0:
                    wav = _match_loudness(wav, ref_rms)
                chunks.append(wav)
                if pbar:
                    pbar.update_absolute(seg_idx + 1, total_segments + 1)

            # 拼接所有段
            final_wav = _concat_audio_segments(chunks, pause_between_segments, sr)
            logger.info(f"LongCat 总时长 {len(final_wav) / sr:.2f}s @ {sr}Hz（{total_segments} 段）")
            result = numpy_audio_to_comfy(final_wav, sr)
            if pbar:
                pbar.update_absolute(total_segments + 1, total_segments + 1)

            return (result,)
        finally:
            try:
                set_unload_mode(unload_mode)
                if unload_mode == "full_unload":
                    unload_model()
                elif unload_mode == "offload_cpu":
                    offload_model_to_cpu()
            finally:
                set_generating(False)

    def _get_model(self, model_path, tokenizer, device, dtype, attention, unload_mode="offload_cpu"):
        key = get_cache_key(model_path, device, dtype, attention, tokenizer)
        cached_model, cached_tokenizer, cached_key = get_cached_model()
        if cached_model is not None and cached_key != key:
            unload_model()
        if cached_model is not None and cached_key == key:
            # 临时设为 keep_gpu 防止 soft_empty_cache 在模型恢复/推理间隙误卸载
            set_unload_mode("keep_gpu")
            if is_offloaded():
                device_str, _ = resolve_device(device)
                resume_model_to_cuda(device_str)
            return cached_model, cached_tokenizer
        model, tokenizer = load_local_model(model_path, device, dtype, attention, tokenizer)
        # 临时设为 keep_gpu 防止 soft_empty_cache 在加载完成到推理之间误卸载
        set_cached_model(model, tokenizer, key, unload_mode="keep_gpu")
        return model, tokenizer
