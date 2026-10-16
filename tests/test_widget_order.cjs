const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
let extension;
vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../web/chinese_labels.js'), 'utf8')
    .replace(/^import .*;\r?\n/, ''), {
    app: { registerExtension(value) { extension = value; } },
    HTMLTextAreaElement: class {},
});
const names = ['model_path', 'tokenizer', 'text', 'prompt_text', 'steps',
    'guidance_strength', 'guidance_method', 'device', 'dtype', 'attention',
    'seed', 'control_after_generate', 'unload_mode', 'pause_between_segments', 'volume_protect'];
const values = ['model', 'tokenizer', 'target', 'reference', 20, 4, 'cfg', 'cuda',
    'bf16', 'sage_attention', 1535347695, 'fixed', 'keep_gpu', 0.3, true];
function createNode() {
    const node = {
        comfyClass: 'LongCatVoiceClone-Huan',
        widgets: names.map(name => ({name, value: null, options: {}})),
        configure(info) {
            this.widgets.forEach((widget, index) => widget.value = info.widgets_values[index]);
        },
    };
    extension.nodeCreated(node);
    return node;
}
const node = createNode();
node.configure({widgets_values: [...values, false]}); // Previous folding switch is ignored.
assert.deepEqual(Object.fromEntries(node.widgets.map(w => [w.name, w.value])),
    Object.fromEntries(names.map((name, i) => [name, values[i]])));
assert.equal(node.widgets[0].name, 'text');
assert.equal(node.widgets.at(-1).name, 'unload_mode');
node.widgets.find(w => w.name === 'steps').value = 24;
const serialized = {widgets_values: node.widgets.map(w => w.value), properties: {huan_advanced: false}};
node.onSerialize(serialized);
const expected = [...values]; expected[4] = 24;
assert.equal(JSON.stringify(serialized.widgets_values), JSON.stringify(expected));
assert.ok(!('huan_advanced' in serialized.properties));
const reloaded = createNode();
reloaded.configure(serialized);
assert.equal(reloaded.widgets.find(w => w.name === 'steps').value, 24);
assert.equal(reloaded.widgets.find(w => w.name === 'seed').value, 1535347695);
console.log('Widget order: legacy loading, editing, serialization, and reload passed.');
