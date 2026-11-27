// 直接在 vm 中驱动真实 AudioWorklet 源码，验证实时分块卷积与参照一致。
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { convolveDirect } from '../public/dsp/convolution.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const workletSrc = readFileSync(path.join(__dirname, '..', 'public', 'worklet', 'convolver-worklet.js'), 'utf8');

function loadProcessorClass() {
  let Captured = null;
  const sandbox = {
    AudioWorkletProcessor: class {
      constructor() {
        this.port = { onmessage: null, postMessage() {} };
      }
    },
    registerProcessor: (_name, cls) => { Captured = cls; },
    sampleRate: 48000,
    console,
    Float32Array,
    Float64Array,
    Math,
  };
  vm.createContext(sandbox);
  vm.runInContext(workletSrc, sandbox);
  assert.equal(typeof Captured, 'function', 'worklet 必须调用 registerProcessor');
  return Captured;
}

test('AudioWorklet 真实源码：128 样本量子块输出 == 直接卷积前缀', () => {
  const Cls = loadProcessorClass();
  const proc = new Cls();
  const h = Float64Array.from({ length: 257 }, (_, i) => 0.5 * Math.cos(i * 0.15) * Math.exp(-i / 80));
  proc.port.onmessage({ data: { type: 'kernel', kernel: Float32Array.from(h) } });

  const B = 128;
  const blocks = 40;
  const x = Float64Array.from({ length: B * blocks }, (_, i) => Math.sin(i * 0.09) * 0.7);
  const ref = convolveDirect(x, h);

  const out = new Float64Array(B * blocks);
  for (let b = 0; b < blocks; b++) {
    const input = new Float32Array(B);
    for (let i = 0; i < B; i++) input[i] = x[b * B + i];
    const outputs = [[new Float32Array(B)]];
    const alive = proc.process([[input]], outputs);
    assert.equal(alive, true);
    for (let i = 0; i < B; i++) out[b * B + i] = outputs[0][0][i];
  }
  let err = 0;
  for (let i = 0; i < out.length; i++) err = Math.max(err, Math.abs(out[i] - ref[i]));
  assert.ok(err < 1e-5, `worklet 与直接卷积误差过大: ${err}`);
});

test('AudioWorklet 未加载核时直通输入', () => {
  const Cls = loadProcessorClass();
  const proc = new Cls();
  const input = new Float32Array(128).fill(0.3);
  const outputs = [[new Float32Array(128)]];
  proc.process([[input]], outputs);
  assert.ok(Math.abs(outputs[0][0][10] - 0.3) < 1e-6);
});

test('AudioWorklet flush 排空完整卷积尾并发出 tail-flushed', () => {
  const Cls = loadProcessorClass();
  const proc = new Cls();
  const messages = [];
  proc.port.postMessage = (message) => messages.push(message);

  const h = new Float64Array(65);
  h[32] = 1;
  proc.port.onmessage({ data: { type: 'kernel', kernel: Float32Array.from(h) } });

  const B = 128;
  const blocks = 4;
  const x = new Float64Array(B * blocks);
  x[400] = 0.9;
  const chunks = [];
  for (let b = 0; b < blocks; b++) {
    const input = new Float32Array(B);
    input.set(x.subarray(b * B, (b + 1) * B));
    const outputs = [[new Float32Array(B)]];
    proc.process([[input]], outputs);
    chunks.push(outputs[0][0]);
  }

  proc.port.onmessage({ data: { type: 'flush' } });
  let flushed = false;
  for (let b = 0; b < 4 && !flushed; b++) {
    const outputs = [[new Float32Array(B)]];
    proc.process([[]], outputs);
    chunks.push(outputs[0][0]);
    flushed = messages.some((m) => m.type === 'tail-flushed');
  }

  assert.equal(flushed, true);
  assert.equal(messages.filter((m) => m.type === 'tail-flushed').length, 1);
  const y = new Float64Array(chunks.length * B);
  chunks.forEach((chunk, i) => y.set(chunk, i * B));
  const ref = convolveDirect(x, h);
  let maxAbs = 0;
  for (let i = 0; i < ref.length; i++) maxAbs = Math.max(maxAbs, Math.abs(y[i] - ref[i]));
  assert.ok(maxAbs < 2e-5, `flush 尾部误差过大: ${maxAbs}`);
  assert.ok(Math.abs(y[432] - 0.9) < 1e-5, '冲激峰位/极性必须保留');
});
