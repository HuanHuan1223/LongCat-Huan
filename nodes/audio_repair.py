"""
音频修复模块
实现音频静音修复和数字误差修复功能，模仿Adobe Audition的修复算法
"""

import torch
import numpy as np
from typing import List, Tuple, Optional, Union
import logging
from .silence_detector import SilenceDetector



class AudioRepairer:
    """音频修复器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.detector = SilenceDetector()
    
    def repair_audio(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        repair_method: str = "缩短静音",
        silence_threshold_db: float = -50.0,
        min_silence_duration_ms: float = 100.0,
        target_silence_duration_ms: float = 50.0,
        remove_digital_errors: bool = True,
        max_error_duration_ms: float = 50.0
    ) -> torch.Tensor:
        """
        执行音频修复
        
        Args:
            waveform: 输入音频波形 (channels, samples)
            sample_rate: 采样率
            repair_method: 修复方式 ("缩短静音" 或 "删除静音")
            silence_threshold_db: 静音检测阈值
            min_silence_duration_ms: 最小静音持续时间
            target_silence_duration_ms: 目标静音长度（缩短静音模式）
            remove_digital_errors: 是否移除数字误差
            max_error_duration_ms: 最大数字误差持续时间
        
        Returns:
            torch.Tensor: 修复后的音频波形
        """
        
        original_shape = waveform.shape
        self.logger.info(f"开始音频修复，输入形状: {original_shape}")
        
        # 检测音频问题
        problems = self.detector.detect_audio_problems(
            waveform, sample_rate, silence_threshold_db, min_silence_duration_ms, max_error_duration_ms
        )
        
        silence_segments = problems['silence_segments']
        digital_errors = problems['digital_errors']
        
        self.logger.info(f"检测到 {len(silence_segments)} 个静音段, {len(digital_errors)} 个数字误差")
        
        # 修复静音
        if silence_segments:
            if repair_method == "删除静音":
                waveform = self._remove_silence(waveform, sample_rate, silence_segments)
            elif repair_method == "缩短静音":
                waveform = self._shorten_silence(
                    waveform, sample_rate, silence_segments, target_silence_duration_ms
                )
            else:
                self.logger.warning(f"未知的修复方式: {repair_method}")
        
        # 移除数字误差
        if remove_digital_errors and digital_errors:
            waveform = self._remove_digital_errors(waveform, sample_rate, digital_errors)
        
        self.logger.info(f"音频修复完成，输出形状: {waveform.shape}")
        return waveform
    
    def _remove_silence(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        silence_segments: List[Tuple[float, float]]
    ) -> torch.Tensor:
        """完全删除静音段"""
        
        # 按时间倒序排序，从后往前删除，避免索引偏移
        silence_segments = sorted(silence_segments, key=lambda x: x[0], reverse=True)
        
        for start_time, end_time in silence_segments:
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            
            # 确保索引有效
            start_sample = max(0, start_sample)
            end_sample = min(waveform.shape[-1], end_sample)
            
            if start_sample < end_sample:
                # 删除静音段
                before_silence = waveform[..., :start_sample]
                after_silence = waveform[..., end_sample:]
                waveform = torch.cat([before_silence, after_silence], dim=-1)
        
        self.logger.info(f"删除了 {len(silence_segments)} 个静音段")
        return waveform
    
    def _shorten_silence(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        silence_segments: List[Tuple[float, float]],
        target_duration_ms: float
    ) -> torch.Tensor:
        """缩短静音段到指定长度"""
        
        target_samples = int(target_duration_ms * sample_rate / 1000)
        
        # 按时间倒序排序，从后往前处理
        silence_segments = sorted(silence_segments, key=lambda x: x[0], reverse=True)
        
        shortened_count = 0
        for start_time, end_time in silence_segments:
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            
            # 确保索引有效
            start_sample = max(0, start_sample)
            end_sample = min(waveform.shape[-1], end_sample)
            
            current_duration_samples = end_sample - start_sample
            
            if current_duration_samples > target_samples:
                # 需要缩短
                before_silence = waveform[..., :start_sample]
                silence_part = waveform[..., start_sample:start_sample + target_samples]
                after_silence = waveform[..., end_sample:]
                
                waveform = torch.cat([before_silence, silence_part, after_silence], dim=-1)
                shortened_count += 1
        
        self.logger.info(f"缩短了 {shortened_count} 个静音段")
        return waveform
    
    def _remove_digital_errors(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        error_segments: List[Tuple[float, float]]
    ) -> torch.Tensor:
        """移除数字音频误差"""
        
        # 按时间倒序排序，从后往前处理
        error_segments = sorted(error_segments, key=lambda x: x[0], reverse=True)
        
        for start_time, end_time in error_segments:
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            
            # 确保索引有效
            start_sample = max(0, start_sample)
            end_sample = min(waveform.shape[-1], end_sample)
            
            if start_sample < end_sample:
                # 使用插值修复替换直接删除
                waveform = self._interpolate_repair(waveform, start_sample, end_sample)
        
        self.logger.info(f"修复了 {len(error_segments)} 个数字误差")
        return waveform
    
    def _interpolate_repair(
        self,
        waveform: torch.Tensor,
        start_sample: int,
        end_sample: int
    ) -> torch.Tensor:
        """使用线性插值修复音频段"""
        
        if start_sample >= end_sample:
            return waveform
        
        # 获取修复段前后的样本值
        if start_sample > 0 and end_sample < waveform.shape[-1]:
            # 前后都有样本，使用线性插值
            start_value = waveform[..., start_sample - 1]
            end_value = waveform[..., end_sample]
            
            # 创建插值
            num_samples = end_sample - start_sample
            interpolation = torch.linspace(0, 1, num_samples, device=waveform.device)
            
            # 计算插值值
            for i in range(num_samples):
                waveform[..., start_sample + i] = start_value + (end_value - start_value) * interpolation[i]
                
        elif start_sample == 0:
            # 开头的错误，用后面的值填充
            if end_sample < waveform.shape[-1]:
                waveform[..., start_sample:end_sample] = waveform[..., end_sample].unsqueeze(-1)
        elif end_sample >= waveform.shape[-1]:
            # 结尾的错误，用前面的值填充
            waveform[..., start_sample:end_sample] = waveform[..., start_sample - 1].unsqueeze(-1)
        
        return waveform


class AdvancedAudioRepairer(AudioRepairer):
    """高级音频修复器，提供更精细的修复控制"""
    
    def repair_with_fade(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        silence_segments: List[Tuple[float, float]],
        fade_duration_ms: float = 10.0
    ) -> torch.Tensor:
        """
        带淡入淡出的静音修复，避免修复后产生咔嚓声
        
        Args:
            fade_duration_ms: 淡入淡出持续时间（毫秒）
        """
        
        fade_samples = int(fade_duration_ms * sample_rate / 1000)
        
        # 按时间倒序排序
        silence_segments = sorted(silence_segments, key=lambda x: x[0], reverse=True)
        
        for start_time, end_time in silence_segments:
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            
            start_sample = max(0, start_sample)
            end_sample = min(waveform.shape[-1], end_sample)
            
            if start_sample < end_sample:
                # 删除静音段
                before_silence = waveform[..., :start_sample]
                after_silence = waveform[..., end_sample:]
                
                # 在连接点添加淡入淡出
                if before_silence.shape[-1] > 0 and after_silence.shape[-1] > 0:
                    # 对before_silence的末尾进行淡出
                    if before_silence.shape[-1] >= fade_samples:
                        fade_out = torch.linspace(1, 0, fade_samples, device=waveform.device)
                        before_silence[..., -fade_samples:] *= fade_out
                    
                    # 对after_silence的开头进行淡入
                    if after_silence.shape[-1] >= fade_samples:
                        fade_in = torch.linspace(0, 1, fade_samples, device=waveform.device)
                        after_silence[..., :fade_samples] *= fade_in
                
                waveform = torch.cat([before_silence, after_silence], dim=-1)
        
        return waveform
    
    def adaptive_silence_repair(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        silence_segments: List[Tuple[float, float]],
        adaptation_factor: float = 0.5
    ) -> torch.Tensor:
        """
        自适应静音修复，根据静音段的长度决定修复策略
        
        Args:
            adaptation_factor: 适应因子，决定缩短比例
        """
        
        repaired_segments = []
        
        for start_time, end_time in silence_segments:
            duration = end_time - start_time
            
            if duration > 1.0:  # 超过1秒的静音，大幅缩短
                target_duration = duration * 0.2
            elif duration > 0.5:  # 0.5-1秒的静音，中等缩短
                target_duration = duration * adaptation_factor
            else:  # 短静音，轻微缩短
                target_duration = duration * 0.8
            
            repaired_segments.append((start_time, start_time + target_duration))
        
        return self._apply_adaptive_repair(waveform, sample_rate, silence_segments, repaired_segments)
    
    def _apply_adaptive_repair(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        original_segments: List[Tuple[float, float]],
        target_segments: List[Tuple[float, float]]
    ) -> torch.Tensor:
        """应用自适应修复"""
        
        # 从后往前处理
        paired_segments = list(zip(original_segments, target_segments))
        paired_segments.sort(key=lambda x: x[0][0], reverse=True)
        
        for (orig_start, orig_end), (target_start, target_end) in paired_segments:
            orig_start_sample = int(orig_start * sample_rate)
            orig_end_sample = int(orig_end * sample_rate)
            target_duration_samples = int((target_end - target_start) * sample_rate)
            
            orig_start_sample = max(0, orig_start_sample)
            orig_end_sample = min(waveform.shape[-1], orig_end_sample)
            
            if orig_start_sample < orig_end_sample and target_duration_samples > 0:
                before_silence = waveform[..., :orig_start_sample]
                silence_part = waveform[..., orig_start_sample:orig_start_sample + target_duration_samples]
                after_silence = waveform[..., orig_end_sample:]
                
                waveform = torch.cat([before_silence, silence_part, after_silence], dim=-1)
        
        return waveform
