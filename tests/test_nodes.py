import importlib.abc
import importlib.util
import json
import math
import sys
import os
import unittest
from unittest.mock import patch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMFY = Path(os.environ.get("COMFYUI_ROOT", ROOT.parent.parent))
sys.path.insert(0, str(COMFY))


class BlockOriginalPlugins(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(token in fullname.lower() for token in ("xiaozhuguang", "huanhuan", "xzg_")):
            raise ImportError(f"Original plugin dependency is forbidden: {fullname}")


sys.meta_path.insert(0, BlockOriginalPlugins())
spec = importlib.util.spec_from_file_location("longcat_huan", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
plugin = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = plugin
spec.loader.exec_module(plugin)

import torch
from longcat_huan.nodes.longcat import _split_text_into_segments
from longcat_huan.nodes.offline_loader import model_names_or_default, tokenizer_names_or_default, resolve_model_path
from longcat_huan.nodes import model_cache
from longcat_huan.nodes.ui_labels import CHOICES, INPUT_LABELS


def audio(samples, rate=24000):
    return {"waveform": samples.reshape(1, 1, -1), "sample_rate": rate}


def tone(duration=1, rate=24000, frequency=440):
    return 0.2 * torch.sin(2 * math.pi * frequency * torch.arange(int(duration * rate)) / rate)


class NodesTest(unittest.TestCase):
    def test_chinese_labels_and_legacy_values(self):
        self.assertEqual(plugin.NODE_DISPLAY_NAME_MAPPINGS["SmartText-Huan"], "数字转中文-Huan")
        clone = plugin.NODE_CLASS_MAPPINGS["LongCatVoiceClone-Huan"]
        schema = clone.INPUT_TYPES()
        for group in schema.values():
            for name, spec in group.items():
                self.assertEqual(spec[1]["display_name"], INPUT_LABELS[name])
                if name in CHOICES:
                    self.assertEqual(spec[0], list(CHOICES[name].values()))
        self.assertTrue(clone.VALIDATE_INPUTS("cfg", "cuda", "bf16", "sdpa", "keep_gpu"))
        self.assertTrue(clone.VALIDATE_INPUTS("标准引导（CFG）", "显卡（CUDA）", "脑浮点16（BF16）", "标准注意力（SDPA）", "保留在显存"))
        self.assertIsInstance(clone.VALIDATE_INPUTS("invalid", "cuda", "bf16", "sdpa", "keep_gpu"), str)
        for cls in plugin.NODE_CLASS_MAPPINGS.values():
            for group in cls.INPUT_TYPES().values():
                for kind, options in group.values():
                    if kind == "BOOLEAN":
                        self.assertEqual((options["label_on"], options["label_off"]), ("开启", "关闭"))


    def setUp(self):
        self.repair = plugin.NODE_CLASS_MAPPINGS["AudioSilenceRepair-Huan"]()

    def test_independent_registration_and_default_style(self):
        self.assertEqual(len(plugin.NODE_CLASS_MAPPINGS), 3)
        self.assertEqual(plugin.WEB_DIRECTORY, "./web")
        self.assertFalse(list((ROOT / "web").rglob("*.css")))
        for name, cls in plugin.NODE_CLASS_MAPPINGS.items():
            self.assertTrue(name.endswith("-Huan"))
            self.assertTrue(plugin.NODE_DISPLAY_NAME_MAPPINGS[name].endswith("-Huan"))
            self.assertEqual(cls.CATEGORY, "LongCat-Huan")
            self.assertFalse(hasattr(cls, "COLOR"))
            self.assertFalse(hasattr(cls, "BGCOLOR"))
            self.assertTrue(callable(getattr(cls, cls.FUNCTION)))
            self.assertIn("required", cls.INPUT_TYPES())

    def test_local_model_discovery(self):
        self.assertIn("LongCat-AudioDiT-3.5B-bf16", model_names_or_default())
        self.assertIn("umt5-base-tokenizer", tokenizer_names_or_default())

    def test_text_conversion(self):
        node = plugin.NODE_CLASS_MAPPINGS["SmartText-Huan"]()
        for text, expected in [("123", "一百二十三"), ("12个", "十二个"), ("12.5元", "十二点五元"), ("12%", "百分之十二"), ("", ""), ("你好", "你好")]:
            with self.subTest(text=text):
                converted, = node.execute(text)
                self.assertEqual(converted, expected)
        self.assertNotEqual(node.IS_CHANGED("123"), node.IS_CHANGED("一百二十三"))
        self.assertEqual(node.RETURN_TYPES, ("STRING",))
        self.assertEqual(node.RETURN_NAMES, ("中文数字文案",))

    def test_long_text_segmentation(self):
        for text in ("你好世界" * 100, "这是第一句话。" * 60, "这是没有句号但是有逗号，" * 50):
            with self.subTest(text=text[:20]):
                segments = _split_text_into_segments(text, 10, 24000, 2048)
                self.assertGreater(len(segments), 1)
                self.assertEqual("".join(s for s, _ in segments), text)
                for sentence, frames in segments:
                    self.assertLessEqual(len(sentence) * 0.21, 10)
                    self.assertLessEqual(frames * 2048 / 24000, 10)

    def test_repair_skip_and_silence_modes(self):
        original = audio(torch.cat([tone(), torch.zeros(24000), tone()]))
        skipped, _ = self.repair.repair_audio(**{"comfyui音频": original, "跳过处理": True})
        self.assertTrue(torch.equal(skipped["waveform"], original["waveform"]))
        for method in ("缩短静音", "删除静音"):
            with self.subTest(method=method):
                result, report = self.repair.repair_audio(**{"comfyui音频": original, "修复方式": method, "静音阈值(dB)": -50, "最小静音时长(ms)": 100, "目标静音长度(ms)": 150})
                self.assertNotIn("失败", report)
                self.assertNotIn("数字", report)
                self.assertEqual(result["sample_rate"], 24000)
                self.assertGreater(result["waveform"].shape[-1], 44000)
                self.assertLessEqual(result["waveform"].shape[-1], 60000)
                self.assertTrue(torch.isfinite(result["waveform"]).all())
        schema = self.repair.INPUT_TYPES()
        self.assertIn("comfyui音频", schema["required"])
        self.assertTrue(all("音频输入" not in group for group in schema.values()))
        self.assertNotIn("移除数字误差", schema["optional"])
        self.assertNotIn("高级模式", schema["optional"])






    def test_audio_connection_and_invalid_input(self):
        sample = str(COMFY / "input/处理好的音色.mp3")
        waveform, rate = self.repair.processor.process_input(sample)
        connected = self.repair.converter.to_comfyui_format(waveform, rate)
        result, _ = self.repair.repair_audio(**{"comfyui音频": connected, "跳过处理": True})
        self.assertGreater(result["waveform"].shape[-1], result["sample_rate"])
        self.assertGreater(float(result["waveform"].abs().max()), 0)
        with self.assertRaises(RuntimeError):
            self.repair.repair_audio(**{"音频输入": "missing-test-file.wav"})
        with self.assertRaises(RuntimeError):
            self.repair.repair_audio(**{"comfyui音频": sample})

    def test_model_path_containment(self):
        with self.assertRaises(ValueError):
            resolve_model_path("../../outside-model")

    def test_clone_failure_clears_generation_state(self):
        node = plugin.NODE_CLASS_MAPPINGS["LongCatVoiceClone-Huan"]()
        with patch.object(node, "_get_model", side_effect=RuntimeError("test load failure")):
            with self.assertRaisesRegex(RuntimeError, "test load failure"):
                node.generate("LongCat-AudioDiT-3.5B-bf16", "umt5-base-tokenizer", "你好", audio(tone()), "你好", 4, 4, "cfg", "cuda", "bf16", "sdpa", 1, "full_unload")
        self.assertFalse(model_cache._generating)

    def test_workflow_migration(self):
        workflow = json.loads((ROOT / "example_workflows/LongCat-Huan-语音克隆.json").read_text(encoding="utf-8"))
        self.assertEqual(set(workflow["extra"]), {"ds", "workflowRendererVersion"})
        self.assertNotIn("AudioSpeed-Huan", {n["type"] for n in workflow["nodes"]})
        ids = {n["id"] for n in workflow["nodes"]}
        by_id = {n["id"]: n for n in workflow["nodes"]}
        for node in workflow["nodes"]:
            if node["type"] in plugin.NODE_CLASS_MAPPINGS:
                self.assertEqual(set(node["properties"]), {"Node name for S&R"})
                self.assertNotIn("color", node)
                self.assertNotIn("bgcolor", node)
        for link in workflow["links"]:
            self.assertIn(link[1], ids)
            self.assertIn(link[3], ids)
            source = by_id[link[1]]["outputs"][link[2]]
            target = by_id[link[3]]["inputs"][link[4]]
            self.assertIn(link[0], source["links"])
            self.assertEqual(target["link"], link[0])
            if by_id[link[1]]["type"] == "SmartText-Huan":
                self.assertEqual(link[2], 0)
                self.assertEqual(len(by_id[link[1]]["outputs"]), 1)
        repair = next(n for n in workflow["nodes"] if n["type"] == "AudioSilenceRepair-Huan")
        self.assertNotIn("音频输入", [i["name"] for i in repair["inputs"]])
        self.assertEqual(len(repair["widgets_values"]), 6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
