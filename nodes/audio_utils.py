"""
音频处理工具模块
提供音频输入处理、格式转换等功能
"""

import torch
import torchaudio
import librosa
import numpy as np
import requests
import tempfile
import os
from pathlib import Path
from typing import Union, Tuple, Dict, Any
from urllib.parse import urlparse


class AudioInputProcessor:
    """音频输入处理器，支持多种输入格式"""
    
    def __init__(self):
        self.supported_formats = {'.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg'}
    
    def process_input(self, audio_input: Any) -> Tuple[torch.Tensor, int]:
        """
        处理各种音频输入格式
        
        Args:
            audio_input: 音频输入，可以是：
                        - URL字符串 (http/https)
                        - 文件路径字符串
                        - ComfyUI音频格式字典 {'waveform': tensor, 'sample_rate': int}
        
        Returns:
            Tuple[torch.Tensor, int]: (音频波形, 采样率)
        """
        if isinstance(audio_input, dict) and 'waveform' in audio_input:
            # ComfyUI标准音频格式
            return self._process_comfyui_format(audio_input)
        elif isinstance(audio_input, str):
            if self._is_url(audio_input):
                # URL输入
                return self._process_url_input(audio_input)
            else:
                # 文件路径输入
                return self._process_file_input(audio_input)
        else:
            raise ValueError(f"不支持的音频输入格式: {type(audio_input)}")
    
    def _process_comfyui_format(self, audio_data: Dict[str, Any]) -> Tuple[torch.Tensor, int]:
        """处理ComfyUI标准音频格式"""
        waveform = audio_data['waveform']
        sample_rate = audio_data['sample_rate']
        
        # 确保tensor格式正确 (channels, samples)
        if isinstance(waveform, torch.Tensor):
            # 如果是3D tensor (batch, channels, samples)，取第一个batch
            if waveform.dim() == 3:
                waveform = waveform[0]
            # 如果是1D tensor，添加channel维度
            elif waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
        else:
            # 转换为tensor
            waveform = torch.tensor(waveform, dtype=torch.float32)
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
        
        return waveform, sample_rate
    
    def _process_url_input(self, url: str) -> Tuple[torch.Tensor, int]:
        """处理URL音频输入"""
        try:
            # 下载音频文件到临时目录
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            # 获取文件扩展名
            parsed_url = urlparse(url)
            file_ext = Path(parsed_url.path).suffix.lower()
            if not file_ext or file_ext not in self.supported_formats:
                file_ext = '.wav'  # 默认扩展名
            
            # 保存到临时文件
            with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as temp_file:
                for chunk in response.iter_content(chunk_size=8192):
                    temp_file.write(chunk)
                temp_path = temp_file.name
            
            try:
                # 加载音频文件
                waveform, sample_rate = self._load_audio_file(temp_path)
                return waveform, sample_rate
            finally:
                # 清理临时文件
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    
        except Exception as e:
            raise RuntimeError(f"URL音频加载失败: {str(e)}")
    
    def _process_file_input(self, file_path: str) -> Tuple[torch.Tensor, int]:
        """处理文件路径音频输入"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"音频文件不存在: {file_path}")
        
        file_ext = Path(file_path).suffix.lower()
        if file_ext not in self.supported_formats:
            raise ValueError(f"不支持的音频格式: {file_ext}")
        
        return self._load_audio_file(file_path)
    
    def _load_audio_file(self, file_path: str) -> Tuple[torch.Tensor, int]:
        """加载音频文件"""
        try:
            # 优先使用torchaudio加载
            waveform, sample_rate = torchaudio.load(file_path)
            return waveform, sample_rate
        except:
            try:
                # 备用方案：使用librosa加载
                waveform, sample_rate = librosa.load(file_path, sr=None, mono=False)
                # 转换为tensor格式
                if waveform.ndim == 1:
                    waveform = waveform[np.newaxis, :]  # 添加channel维度
                waveform = torch.from_numpy(waveform).float()
                return waveform, sample_rate
            except Exception as e:
                raise RuntimeError(f"音频文件加载失败: {str(e)}")
    
    def _is_url(self, string: str) -> bool:
        """检查字符串是否为URL"""
        try:
            result = urlparse(string)
            return all([result.scheme, result.netloc]) and result.scheme in ['http', 'https']
        except:
            return False


class AudioFormatConverter:
    """音频格式转换器"""
    
    @staticmethod
    def to_comfyui_format(waveform: torch.Tensor, sample_rate: int) -> Dict[str, Any]:
        """转换为ComfyUI标准格式"""
        # 确保是3D tensor格式 (batch, channels, samples)
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)  # 添加batch维度
        elif waveform.dim() == 1:
            waveform = waveform.unsqueeze(0).unsqueeze(0)  # 添加batch和channel维度
        
        return {
            'waveform': waveform,
            'sample_rate': sample_rate
        }
    
    @staticmethod
    def to_mono(waveform: torch.Tensor) -> torch.Tensor:
        """转换为单声道"""
        if waveform.shape[0] > 1:
            # 多声道转单声道：取平均值
            return waveform.mean(dim=0, keepdim=True)
        return waveform
    
    @staticmethod
    def resample(waveform: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
        """重采样音频"""
        if orig_sr == target_sr:
            return waveform
        
        # 使用torchaudio进行重采样
        resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
        return resampler(waveform)
