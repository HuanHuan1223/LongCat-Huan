"""Three independent ComfyUI nodes using the default ComfyUI interface."""
from .nodes.longcat import LongCatVoiceCloneHuan
from .nodes.text import SmartTextHuan
from .nodes.audio_diagnostic_repair import AudioDiagnosticRepair

NODE_CLASS_MAPPINGS = {
    "LongCatVoiceClone-Huan": LongCatVoiceCloneHuan,
    "SmartText-Huan": SmartTextHuan,
    "AudioSilenceRepair-Huan": AudioDiagnosticRepair,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LongCatVoiceClone-Huan": "LongCat语音克隆-Huan",
    "SmartText-Huan": "数字转中文-Huan",
    "AudioSilenceRepair-Huan": "音频静音修复-Huan",
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
