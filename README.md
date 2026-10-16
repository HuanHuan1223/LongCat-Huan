# LongCat-Huan

**让数字更好读，让长文案也能用熟悉的声音讲出来。**

LongCat-Huan 是一款面向中文配音的 ComfyUI 插件，将 **声音克隆、数字转中文、长文本分段合成和静音修复** 串成一套工作流。提供一段参考音频和需要朗读的文案，就能生成参考音色的配音，适合口播、故事、文章朗读和视频解说。

## 工作流预览

![LongCat-Huan 语音克隆工作流](docs/images/workflow.png)

[查看与下载示例工作流](example_workflows/LongCat-Huan-语音克隆.json)

## 插件有哪些优势？

### 数字先转成中文，减少配音读错

文案里的年份、金额、小数和百分比，直接交给语音模型时容易出现读法不符合预期的问题。**数字转中文-Huan** 会先把这些内容转换成适合朗读的中文，再交给语音克隆节点，省去逐个手动改写数字的麻烦。

| 原始文案 | 转换后的朗读文案 |
| --- | --- |
| 2026年 | 二零二六年 |
| 12.5元 | 十二点五元 |
| 35% | 百分之三十五 |
| 12个 | 十二个 |
| 第12章 | 第十二章 |

支持常见数字、日期、时间、比例和单位等表达。转换结果可以先预览、再生成配音；特殊编号或有多种读法的内容，也可以按需要手动调整。

### 支持长文本，自动分段并合成完整音频

长篇文案可以直接输入，插件会根据文本和时长估算自动分段，逐段生成后拼接成一条音频，无需自己拆成多个语音节点。

- 适合故事、文章朗读、长口播和解说文案。
- 支持调节 **段间停顿**，控制各段之间的间隔。
- 长句也会进一步拆分，便于处理较长的连续内容。

实际生成时间取决于文案长度、模型设置和设备性能。

### 参考音色克隆，配音内容自由替换

上传参考音频，填写音频里实际说出的文字，再输入新的文案，即可按参考音色生成语音。更换待生成文案，就能继续制作同一音色的其他内容。

支持 **匹配参考音量**，让生成音频的音量更接近参考素材。清晰、干净的参考音频和准确的参考文字，有助于获得更稳定的效果。

### 缩短多余静音，让朗读更紧凑

**音频静音修复-Huan** 可检测音频中的静音片段，并选择缩短或删除。对于生成结果中偏长的停顿，可以按需要调整，减少后期手动剪掉空白的工作。

支持设置静音阈值、最小静音时长和目标静音长度，也可开启 **跳过处理** 对比原始效果。

### 中文界面，操作直观

三个节点的参数与接口均采用中文显示，沿用 ComfyUI 默认界面。文案使用大文本框，常用参数集中排列，直接连接节点即可使用。

## 包含哪些节点？

| 节点 | 用途 |
| --- | --- |
| **LongCat语音克隆-Huan** | 参考音色克隆、长文本自动分段与拼接、段间停顿、参考音量匹配 |
| **数字转中文-Huan** | 将文案中的数字转换成中文朗读形式，输出“中文数字文案” |
| **音频静音修复-Huan** | 缩短或删除静音片段，输出修复后的音频与诊断报告 |

## 安装

1. 下载本仓库，将插件文件夹命名为 `LongCat-Huan`，放入 `ComfyUI/custom_nodes/`。
2. 使用运行 ComfyUI 的 Python 环境安装 `requirements.txt` 中的依赖。
3. 将语音模型和文本分词器放入下方目录。
4. 重启 ComfyUI，刷新页面，搜索 `Huan` 即可找到节点。

也可以在 `ComfyUI/custom_nodes/` 目录中执行：

```bash
git clone https://github.com/HuanHuan1223/LongCat-Huan.git
```

### 模型下载

示例工作流使用以下两个下载项目，分别放入对应目录：

| 下载项目 | 下载地址 | 保存目录（相对于 ComfyUI） |
| --- | --- | --- |
| LongCat-AudioDiT-3.5B-bf16 语音模型 | [打开模型下载页](https://huggingface.co/drbaph/LongCat-AudioDiT-3.5B-bf16/tree/main) | `models/audiodit/LongCat-AudioDiT-3.5B-bf16/` |
| UMT5 文本分词器 | [打开分词器下载页](https://huggingface.co/google/umt5-base/tree/main) | `models/audiodit/umt5-base-tokenizer/` |

**语音模型下载这两个文件**，保持文件名不变：

- [model.safetensors（约 7.67 GB）](https://huggingface.co/drbaph/LongCat-AudioDiT-3.5B-bf16/resolve/main/model.safetensors?download=true)
- [config.json](https://huggingface.co/drbaph/LongCat-AudioDiT-3.5B-bf16/resolve/main/config.json?download=true)

**文本分词器下载以下四个文件**，放在同一个文件夹：

- [tokenizer_config.json](https://huggingface.co/google/umt5-base/resolve/main/tokenizer_config.json?download=true)
- [special_tokens_map.json](https://huggingface.co/google/umt5-base/resolve/main/special_tokens_map.json?download=true)
- [spiece.model](https://huggingface.co/google/umt5-base/resolve/main/spiece.model?download=true)
- [tokenizer.json](https://huggingface.co/google/umt5-base/resolve/main/tokenizer.json?download=true)

分词器文件合计约 22 MB，无需下载该页面中的 `pytorch_model.bin`。下载完成后重启 ComfyUI，在节点中选择 `LongCat-AudioDiT-3.5B-bf16` 和 `umt5-base-tokenizer`。

### 模型放置位置

模型文件需单独准备，不包含在插件下载包内。

```text
ComfyUI/models/audiodit/
├── LongCat-AudioDiT-3.5B-bf16/
│   ├── config.json
│   └── model.safetensors
└── umt5-base-tokenizer/
    ├── tokenizer_config.json
    ├── special_tokens_map.json
    ├── spiece.model
    └── tokenizer.json
```

## 使用方法

1. **导入工作流**：打开 `example_workflows/LongCat-Huan-语音克隆.json`。
2. **上传参考音频**：在“加载音频”节点选择自己的参考素材。
3. **填写参考文字**：在克隆节点的“参考音频对应文字”中填写音频实际说出的内容，换音频时一起更新。
4. **输入配音文案**：将需要朗读的内容放入“数字转中文-Huan”，转换后的输出连接克隆节点的“待生成文案”。
5. **调整停顿**：按需要设置段间停顿，并选择是否进行静音修复。
6. **运行与试听**：执行工作流，在“预览音频”节点中试听生成结果。

示例工作流中的参考音频文件名只是示例，导入后请重新选择自己的文件。模型、运行设备和注意力算法也应按本机环境选择；没有安装 SageAttention 时，可改用标准注意力（SDPA）。

> 小提示：长文案可以先截取一小段试听，确认音色、数字读法和停顿合适后，再生成全文。
