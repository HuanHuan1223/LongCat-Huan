"""
ComfyUI音频诊断修复节点
模仿Adobe Audition的音频诊断修复功能
"""

import torch
import logging
from typing import Any, Dict, Union

from .audio_utils import AudioInputProcessor, AudioFormatConverter
from .silence_detector import SilenceDetector
from .audio_repair import AudioRepairer
from .ui_labels import chinese_inputs



class AudioDiagnosticRepair:
    """音频静音修复-Huan节点"""
    
    def __init__(self):
        self.processor = AudioInputProcessor()
        self.converter = AudioFormatConverter()
        self.detector = SilenceDetector()
        self.repairer = AudioRepairer()
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    @classmethod
    def INPUT_TYPES(cls):
        return chinese_inputs({
            "required": {
                "comfyui音频": ("AUDIO",),
                "跳过处理": ("BOOLEAN", {
                    "default": False
                }),
                "修复方式": (["缩短静音", "删除静音"], {
                    "default": "缩短静音"
                }),
                "静音阈值(dB)": ("FLOAT", {
                    "default": -50.0,
                    "min": -80.0,
                    "max": -20.0,
                    "step": 1.0,
                    "display": "slider"
                }),
                "最小静音时长(ms)": ("FLOAT", {
                    "default": 100.0,
                    "min": 10.0,
                    "max": 1000.0,
                    "step": 10.0,
                    "display": "slider"
                }),
            },
            "optional": {
                "目标静音长度(ms)": ("FLOAT", {
                    "default": 50.0,
                    "min": 0.0,
                    "max": 500.0,
                    "step": 10.0,
                    "display": "slider"
                }),
                "淡入淡出(ms)": ("FLOAT", {
                    "default": 10.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 1.0,
                    "display": "slider"
                })
            }
        })
    
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("修复后音频", "诊断报告")
    FUNCTION = "repair_audio"
    CATEGORY = "LongCat-Huan"
    DESCRIPTION = "检测静音片段，按目标长度缩短静音或删除静音。"
    
    
    def repair_audio(self, **kwargs):
        """执行音频诊断修复"""
        try:
            # 获取参数，使用kwargs.get()确保兼容性
            skip_processing = kwargs.get("跳过处理", False)
            comfyui_audio = kwargs.get("comfyui音频", None)
            repair_method = kwargs.get("修复方式", "缩短静音")
            silence_threshold = kwargs.get("静音阈值(dB)", -50.0)
            min_silence_duration = kwargs.get("最小静音时长(ms)", 100.0)
            target_silence_duration = kwargs.get("目标静音长度(ms)", 50.0)

            # 确定音频输入源
            if not isinstance(comfyui_audio, dict):
                raise ValueError("请连接ComfyUI音频输出到音频输入接口")

            # 处理音频输入
            waveform, sample_rate = self.processor.process_input(comfyui_audio)
            self.logger.info(f"音频加载成功: 形状={waveform.shape}, 采样率={sample_rate}")

            # 如果跳过处理，直接返回原音频
            if skip_processing:
                self.logger.info("跳过处理: 直接返回原音频")
                output_audio = self.converter.to_comfyui_format(waveform, sample_rate)
                return (output_audio, "【已跳过处理】音频未做任何修改直接输出")
            
            # 执行诊断
            silence_segments = self.detector.detect_silence(
                waveform, sample_rate, silence_threshold, min_silence_duration
            )
            problems = {'silence_segments': silence_segments, 'total_issues': len(silence_segments)}

            report = self._generate_diagnostic_report(problems, waveform.shape[-1], sample_rate)
            
            # 执行修复
            if problems['total_issues'] > 0:
                if repair_method == "删除静音":
                    repaired_waveform = self.repairer._remove_silence(waveform, sample_rate, silence_segments)
                else:
                    repaired_waveform = self.repairer._shorten_silence(
                        waveform, sample_rate, silence_segments, target_silence_duration
                    )

                original_duration = waveform.shape[-1] / sample_rate
                repaired_duration = repaired_waveform.shape[-1] / sample_rate
                time_saved = original_duration - repaired_duration
                report += f"\n\n修复完成！\n原始时长: {original_duration:.2f}秒\n修复后时长: {repaired_duration:.2f}秒\n节省时间: {time_saved:.2f}秒"
                
            else:
                repaired_waveform = waveform
                report += "\n\n未检测到需要修复的问题，音频保持原样。"
            
            # 转换为ComfyUI格式输出
            output_audio = self.converter.to_comfyui_format(repaired_waveform, sample_rate)
            
            self.logger.info("音频诊断修复完成")
            return (output_audio, report)
            
        except Exception as e:
            error_msg = f"音频诊断修复失败: {str(e)}"
            self.logger.error(error_msg)
            
            raise RuntimeError(error_msg) from e
    

    def _generate_diagnostic_report(self, problems: dict, total_samples: int, sample_rate: int) -> str:
        """生成诊断报告"""
        total_duration = total_samples / sample_rate
        silence_segments = problems.get('silence_segments', [])
        
        report = "🔍 音频诊断报告\n"
        report += "=" * 30 + "\n"
        report += f"总时长: {total_duration:.2f}秒\n"
        report += f"采样率: {sample_rate}Hz\n\n"
        
        # 静音段报告
        if silence_segments:
            total_silence_duration = sum(end - start for start, end in silence_segments)
            silence_percentage = (total_silence_duration / total_duration) * 100
            
            report += f"🔇 检测到 {len(silence_segments)} 个静音段:\n"
            report += f"静音总时长: {total_silence_duration:.2f}秒 ({silence_percentage:.1f}%)\n"
            
            # 显示前5个静音段详情
            for i, (start, end) in enumerate(silence_segments[:5]):
                duration = end - start
                report += f"  {i+1}. {start:.2f}s - {end:.2f}s (时长: {duration:.2f}s)\n"
            
            if len(silence_segments) > 5:
                report += f"  ... 还有 {len(silence_segments) - 5} 个静音段\n"
        else:
            report += "🔇 未检测到明显静音段\n"
        
        report += "\n"
        
        # 总结
        total_issues = len(silence_segments)
        if total_issues == 0:
            report += "\n🎉 音频质量良好，无需修复！"
        else:
            report += f"\n📋 共发现 {total_issues} 个问题待修复"
        
        return report
