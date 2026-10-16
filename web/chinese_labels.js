import { app } from "../../scripts/app.js";

const seedModes = {
    fixed: "保持不变",
    increment: "每次递增",
    decrement: "每次递减",
    randomize: "每次随机",
};

const displayOrder = [
    "text", "prompt_text", "pause_between_segments", "volume_protect",
    "seed", "control_after_generate", "steps", "guidance_strength", "guidance_method",
    "model_path", "tokenizer", "device", "dtype", "attention", "unload_mode",
];

function arrangeWidgets(node) {
    // Workflow arrays retain their original order; only the visible widgets move.
    const original = [...node.widgets];
    node.widgets.sort((a, b) => displayOrder.indexOf(a.name) - displayOrder.indexOf(b.name));
    const configure = node.configure;
    node.configure = function (info) {
        const values = info.widgets_values;
        const remapped = Array.isArray(values)
            ? this.widgets.map(widget => values[original.indexOf(widget)]) : values;
        return configure.call(this, { ...info, widgets_values: remapped });
    };
    const serialize = node.onSerialize;
    node.onSerialize = function (info) {
        serialize?.call(this, info);
        const values = info.widgets_values;
        if (Array.isArray(values)) {
            info.widgets_values = original.map(widget => values[this.widgets.indexOf(widget)]);
        }
        delete info.properties?.huan_advanced;
    };
}

app.registerExtension({
    name: "LongCat-Huan.ChineseLabels",
    nodeCreated(node) {
        if (node.comfyClass !== "LongCatVoiceClone-Huan") return;
        const control = node.widgets?.find(widget => widget.name === "control_after_generate");
        if (control) {
            // Native combo label formatting keeps the original serialized seed modes.
            control.options.getOptionLabel = value => seedModes[value] ?? value ?? "";
        }
        for (const [name, title, placeholder, height] of [
            ["text", "待生成文案", "填写希望用克隆音色朗读的文案，支持多行长文本", 260],
            ["prompt_text", "参考音频对应文字", "填写参考音频中实际说出的原文，换音频时请一起更新", 140],
        ]) {
            const widget = node.widgets?.find(widget => widget.name === name);
            if (!(widget?.element instanceof HTMLTextAreaElement)) continue;
            const textarea = widget.element;
            const container = document.createElement("label");
            container.style.cssText = "display:flex;flex-direction:column;height:100%;gap:4px;color:var(--input-text);font-size:12px";
            const caption = document.createElement("span");
            caption.textContent = title;
            textarea.placeholder = placeholder;
            textarea.setAttribute("aria-label", title);
            textarea.style.minHeight = "0";
            textarea.style.flex = "1";
            container.append(caption, textarea);
            widget.element = container;
            widget.options.getMinHeight = () =>
                node.inputs?.some(input => input.name === name && input.link != null) ? 0 : height;
        }
        const drawForeground = node.onDrawForeground;
        node.onDrawForeground = function (ctx, canvas, ...args) {
            drawForeground?.call(this, ctx, canvas, ...args);
            if (this.flags.collapsed || canvas.low_quality) return;
            const index = this.inputs?.findIndex(input => input.name === "text" && input.link != null);
            if (index == null || index < 0) return;
            this.arrange();
            const position = this.getInputPos(index);
            ctx.save();
            ctx.font = canvas.inner_text_font;
            ctx.fillStyle = LiteGraph.NODE_TEXT_COLOR;
            ctx.textAlign = "left";
            ctx.fillText("待生成文案", position[0] - this.pos[0] + 10, position[1] - this.pos[1] + 4);
            ctx.restore();
        };
        arrangeWidgets(node);
    },
    loadedGraphNode(node) {
        if (node.comfyClass === "LongCatVoiceClone-Huan") {
            delete node.properties?.huan_advanced;
            node.setSize([node.size[0], node.computeSize()[1]]);
            for (const input of node.inputs ?? []) {
                if (input.name === "text") input.label = "待生成文案";
                if (input.name === "prompt_audio") input.label = "参考音频";
            }
        }
        if (node.comfyClass === "SmartText-Huan" && node.title === "智能文本框-Huan") {
            node.title = "数字转中文-Huan";
        }
    },
});
