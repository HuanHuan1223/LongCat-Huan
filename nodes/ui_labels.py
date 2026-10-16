"""Chinese labels for native ComfyUI widgets, with legacy value support."""

INPUT_LABELS = {
    "model_path": "语音模型", "tokenizer": "文本分词器", "text": "待生成文案",
    "prompt_audio": "参考音频", "prompt_text": "参考音频对应文字",
    "steps": "采样步数", "guidance_strength": "引导强度",
    "guidance_method": "引导方式", "device": "运行设备", "dtype": "计算精度",
    "attention": "注意力算法", "seed": "随机种子", "unload_mode": "模型卸载方式",
    "pause_between_segments": "段间停顿（秒）", "volume_protect": "匹配参考音量",
    "comfyui音频": "音频输入接口",
}

CHOICES = {
    "guidance_method": {"cfg": "标准引导（CFG）", "apg": "自适应引导（APG）"},
    "device": {"auto": "自动选择", "cuda": "显卡（CUDA）", "cpu": "处理器（CPU）", "mps": "苹果显卡（MPS）"},
    "dtype": {"auto": "自动精度", "bf16": "脑浮点16（BF16）", "fp16": "半精度（FP16）", "fp32": "单精度（FP32）"},
    "attention": {"auto": "自动选择", "sdpa": "标准注意力（SDPA）", "sage_attention": "Sage加速注意力", "flash_attention": "Flash加速注意力"},
    "unload_mode": {"full_unload": "完全卸载模型", "offload_cpu": "转移到内存", "keep_gpu": "保留在显存"},
    "算法选择": {"simple": "简易变速", "phase_vocoder": "相位声码器", "wsola": "波形相似叠加（WSOLA）"},
    "处理质量": {"high": "高质量", "medium": "标准质量", "low": "快速处理"},
}


def chinese_inputs(schema):
    for group in schema.values():
        for name, spec in group.items():
            kind, *rest = spec
            options = dict(rest[0]) if rest else {}
            if name in INPUT_LABELS:
                options["display_name"] = INPUT_LABELS[name]
            if kind == "BOOLEAN":
                options.update(label_on="开启", label_off="关闭")
            if name in CHOICES:
                choices = CHOICES[name]
                kind = [choices[value] for value in kind]
                if "default" in options:
                    options["default"] = choices[options["default"]]
            group[name] = (kind, options)
    return schema


def internal_choice(name, value):
    choices = CHOICES[name]
    if value in choices:
        return value
    for internal, label in choices.items():
        if value == label:
            return internal
    raise ValueError(f"{INPUT_LABELS.get(name, name)}的选项无效：{value}")


def validate_choices(values):
    try:
        for name, value in values.items():
            internal_choice(name, value)
    except ValueError as error:
        return str(error)
    return True
