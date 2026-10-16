"""
音频静音检测模块
实现基于阈值的静音检测算法，模仿Adobe Audition的静音检测功能
"""

import torch
import numpy as np
from typing import List, Tuple, Optional
import logging


class SilenceDetector:
    """静音检测器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def detect_silence(
        self, 
        waveform: torch.Tensor, 
        sample_rate: int,
        silence_threshold_db: float = -50.0,
        min_silence_duration_ms: float = 100.0,
        hop_length_ms: float = 10.0
    ) -> List[Tuple[float, float]]:
        """
        检测音频中的静音段
        
        Args:
            waveform: 音频波形 (channels, samples)
            sample_rate: 采样率
            silence_threshold_db: 静音阈值 (dB)
            min_silence_duration_ms: 最小静音持续时间 (毫秒)
            hop_length_ms: 检测窗口跳跃长度 (毫秒)
        
        Returns:
            List[Tuple[float, float]]: 静音段列表 [(开始时间, 结束时间), ...]
        """
        
        # 转换为单声道
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0)
        else:
            waveform = waveform.squeeze(0)
        
        # 计算参数
        hop_length = int(hop_length_ms * sample_rate / 1000)
        min_silence_samples = int(min_silence_duration_ms * sample_rate / 1000)
        
        # 计算RMS能量
        rms_values = self._calculate_rms_energy(waveform, hop_length)
        
        # 转换为dB
        db_values = self._rms_to_db(rms_values)
        
        # 检测静音段
        silence_mask = db_values < silence_threshold_db
        silence_segments = self._find_continuous_segments(silence_mask, hop_length, sample_rate)
        
        # 过滤短静音段
        filtered_segments = [
            (start, end) for start, end in silence_segments 
            if (end - start) >= (min_silence_duration_ms / 1000.0)
        ]
        
        self.logger.info(f"检测到 {len(filtered_segments)} 个静音段")
        return filtered_segments
    
    def detect_digital_errors(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        max_error_duration_ms: float = 50.0,
        amplitude_threshold: float = 0.8
    ) -> List[Tuple[float, float]]:
        """
        检测短暂的数字音频误差（如爆音、咔嚓声等）
        
        Args:
            waveform: 音频波形
            sample_rate: 采样率
            max_error_duration_ms: 最大误差持续时间 (毫秒)
            amplitude_threshold: 振幅阈值
        
        Returns:
            List[Tuple[float, float]]: 数字误差段列表
        """
        
        # 转换为单声道
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0)
        else:
            waveform = waveform.squeeze(0)
        
        # 检测突然的振幅变化
        diff = torch.abs(torch.diff(waveform))
        
        # 查找超过阈值的点
        error_points = torch.where(diff > amplitude_threshold)[0]
        
        if len(error_points) == 0:
            return []
        
        # 将相近的点归组为错误段
        max_error_samples = int(max_error_duration_ms * sample_rate / 1000)
        error_segments = []
        
        current_start = error_points[0].item()
        current_end = current_start
        
        for point in error_points[1:]:
            point = point.item()
            if point - current_end <= max_error_samples:
                current_end = point
            else:
                # 当前段结束，开始新段
                start_time = current_start / sample_rate
                end_time = (current_end + 1) / sample_rate
                if (end_time - start_time) <= (max_error_duration_ms / 1000.0):
                    error_segments.append((start_time, end_time))
                
                current_start = point
                current_end = point
        
        # 处理最后一段
        start_time = current_start / sample_rate
        end_time = (current_end + 1) / sample_rate
        if (end_time - start_time) <= (max_error_duration_ms / 1000.0):
            error_segments.append((start_time, end_time))
        
        self.logger.info(f"检测到 {len(error_segments)} 个数字误差")
        return error_segments
    
    def detect_audio_problems(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        silence_threshold_db: float = -50.0,
        min_silence_duration_ms: float = 100.0,
        max_error_duration_ms: float = 50.0
    ) -> dict:
        """
        综合音频问题检测
        
        Returns:
            dict: 包含静音段和数字误差的字典
        """
        
        silence_segments = self.detect_silence(
            waveform, sample_rate, silence_threshold_db, min_silence_duration_ms
        )
        
        digital_errors = self.detect_digital_errors(
            waveform, sample_rate, max_error_duration_ms
        )
        
        return {
            'silence_segments': silence_segments,
            'digital_errors': digital_errors,
            'total_issues': len(silence_segments) + len(digital_errors)
        }
    
    def _calculate_rms_energy(self, waveform: torch.Tensor, hop_length: int) -> torch.Tensor:
        """计算RMS能量"""
        # 分帧
        frames = waveform.unfold(0, hop_length, hop_length)
        # 计算每帧的RMS
        rms = torch.sqrt(torch.mean(frames ** 2, dim=1))
        return rms
    
    def _rms_to_db(self, rms_values: torch.Tensor, reference: float = 1.0) -> torch.Tensor:
        """将RMS值转换为dB"""
        # 避免对0取对数
        rms_values = torch.clamp(rms_values, min=1e-8)
        return 20 * torch.log10(rms_values / reference)
    
    def _find_continuous_segments(
        self, 
        mask: torch.Tensor, 
        hop_length: int, 
        sample_rate: int
    ) -> List[Tuple[float, float]]:
        """查找连续的True段"""
        # 找到状态变化点
        mask_np = mask.numpy()
        changes = np.diff(np.concatenate(([False], mask_np, [False])).astype(int))
        
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]
        
        # 转换为时间
        segments = []
        for start, end in zip(starts, ends):
            start_time = start * hop_length / sample_rate
            end_time = end * hop_length / sample_rate
            segments.append((start_time, end_time))
        
        return segments


class AdvancedSilenceDetector(SilenceDetector):
    """高级静音检测器，提供更精确的检测算法"""
    
    def detect_silence_advanced(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        silence_threshold_db: float = -50.0,
        min_silence_duration_ms: float = 100.0,
        spectral_threshold: float = 0.1
    ) -> List[Tuple[float, float]]:
        """
        基于能量和频谱特征的高级静音检测
        
        Args:
            spectral_threshold: 频谱阈值，用于更精确的静音判断
        """
        
        # 基础能量检测
        energy_silences = self.detect_silence(
            waveform, sample_rate, silence_threshold_db, min_silence_duration_ms
        )
        
        # 转换为单声道
        if waveform.shape[0] > 1:
            mono_waveform = waveform.mean(dim=0)
        else:
            mono_waveform = waveform.squeeze(0)
        
        # 计算短时傅里叶变换
        n_fft = 2048
        hop_length = n_fft // 4
        
        # 计算频谱质心
        stft = torch.stft(
            mono_waveform, 
            n_fft=n_fft, 
            hop_length=hop_length, 
            return_complex=True
        )
        magnitude = torch.abs(stft)
        
        # 计算频谱质心
        freqs = torch.fft.fftfreq(n_fft, 1/sample_rate)[:n_fft//2+1]
        spectral_centroids = torch.sum(magnitude * freqs.unsqueeze(1), dim=0) / (torch.sum(magnitude, dim=0) + 1e-8)
        
        # 归一化频谱质心
        normalized_centroids = spectral_centroids / (sample_rate / 2)
        
        # 基于频谱特征进一步筛选静音段
        refined_silences = []
        for start_time, end_time in energy_silences:
            start_frame = int(start_time * sample_rate / hop_length)
            end_frame = int(end_time * sample_rate / hop_length)
            
            if start_frame < len(normalized_centroids) and end_frame <= len(normalized_centroids):
                avg_centroid = torch.mean(normalized_centroids[start_frame:end_frame])
                
                # 如果频谱质心很低，更可能是真正的静音
                if avg_centroid < spectral_threshold:
                    refined_silences.append((start_time, end_time))
        
        self.logger.info(f"高级检测精化后剩余 {len(refined_silences)} 个静音段")
        return refined_silences
