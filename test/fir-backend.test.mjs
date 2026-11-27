import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import { fileURLToPath } from 'node:url';

import { convolveDirect } from '../public/dsp/convolution.js';
import { designFir } from '../public/dsp/fir-design.js';
import {
  DeferredTailConvolutionBackend,
  MergedFinalBlockConvolutionBackend,
  FirBackendAdapter,
  HotSwapConvolutionRecorder,
  convolveSwitchedDirect,
  renderSwitchedWithBackend,
  convolutionErrorMetrics,
  createCancellationTailProbe,
  inspectTailBeats,
  differenceToOriginalTrack,
  migrateFilterConfig,
  loadMigratedConfigAndProcess,
  verifyFirBackend,
  TOLERANCE_PROFILES,
} from '../public/dsp/fir-backend.js';

const points = [
  { f: 0, db: 0 }, { f: 0.2, db: -1 }, { f: 0.45, db: -12 }, { f: 1, db: -40 },
];

function kernel(N = 255) {
  const d = designFir({ points, N, firType: 'I', window: 'hann' });
  assert.equal(d.ok, true, d.message);
  return d.kernel;
}

test('两个后端报告相同整块延迟，但尾部冲洗约定不同', () => {
  const h = kernel(127);
  const a = new DeferredTailConvolutionBackend(h, { blockSize: 128 });
  const b = new MergedFinalBlockConvolutionBackend(h, { blockSize: 128 });
  assert.equal(a.wholeBlockLatencySamples, b.wholeBlockLatencySamples);
  assert.equal(a.latencyReport().wholeBlockLatencySamples, 128);
  assert.equal(b.latencyReport().wholeBlockLatencySamples, 128);
  assert.equal(a.tailConvention, 'deferred-drain');
  assert.equal(b.tailConvention, 'merged-final');
  assert.equal(typeof a.drain, 'function');
  assert.equal(typeof b.drain, 'undefined');
});

for (const N of [65, 127, 255]) {
  for (const B of [1, 127, 128, 256]) {
    test(`双尾部后端与直接卷积一致 N=${N} B=${B}`, () => {
      const h = kernel(N);
      const x = Float64Array.from({ length: 1337 }, (_, i) => Math.sin(i * 0.083) + 0.2 * Math.cos(i * 0.021));
      const ref = convolveDirect(x, h);
      for (const Backend of [DeferredTailConvolutionBackend, MergedFinalBlockConvolutionBackend]) {
        const y = new Backend(h, { blockSize: B }).processAll(x);
        assert.equal(y.length, x.length + h.length - 1);
        const err = convolutionErrorMetrics(y, ref);
        assert.equal(err.pass, true, `${Backend.name}: max=${err.maxAbs} len=${y.length}/${ref.length}`);
      }
    });
  }
}

test('长滤波器和长输入结束仍保留完整尾部', () => {
  const h = kernel(4097);
  const x = Float64Array.from({ length: 12000 }, (_, i) => 0.3 * Math.sin(i * 0.019) + 0.1 * Math.sin(i * 0.41));
  const ref = convolveDirect(x, h);
  const evidence = verifyFirBackend({ long: true, N: 4097, inputLength: 12000 });
  assert.equal(evidence.pass, true, evidence.checks.filter((c) => !c.pass).map((c) => c.id).join(','));
  for (const Backend of [DeferredTailConvolutionBackend, MergedFinalBlockConvolutionBackend]) {
    const y = new Backend(h, { blockSize: 128 }).processAll(x);
    assert.equal(y.length, ref.length);
    assert.ok(convolutionErrorMetrics(y, ref).maxAbs < 1e-8);
  }
});

test('适配器自动识别 deferred/merged，且不会把尾部记两次', () => {
  const h = kernel(127);
  const x = Float64Array.from({ length: 900 }, (_, i) => Math.sin(i * 0.17));
  const ref = convolveDirect(x, h);

  const deferred = new FirBackendAdapter(new DeferredTailConvolutionBackend(h), { kernel: h });
  deferred.push(x.subarray(0, 300));
  deferred.push(x.subarray(300));
  deferred.end();
  const y1 = deferred.output();
  assert.equal(y1.length, ref.length);
  assert.ok(convolutionErrorMetrics(y1, ref).pass);
  assert.deepEqual(deferred.tailCalls, ['end', 'drain']);

  const streaming = new FirBackendAdapter(new DeferredTailConvolutionBackend(h), { kernel: h });
  const headA = streaming.push(x.subarray(0, 300));
  const headB = streaming.push(x.subarray(300));
  const lastBlock = streaming.end();
  const drainedTail = streaming.drain();
  const streamed = Float64Array.from([...headA, ...headB, ...lastBlock, ...drainedTail]);
  assert.equal(drainedTail.length, h.length - 1);
  assert.equal(streamed.length, ref.length);
  assert.ok(convolutionErrorMetrics(streamed, ref).pass);

  const merged = new FirBackendAdapter(new MergedFinalBlockConvolutionBackend(h), { kernel: h });
  merged.push(x);
  merged.end();
  const y2 = merged.output();
  assert.equal(y2.length, ref.length);
  assert.ok(convolutionErrorMetrics(y2, ref).pass);
  assert.deepEqual(merged.tailCalls, ['finish']);
});

test('末块反相探针只数到两枚尾拍；重复尾部会被检出', () => {
  const probe = createCancellationTailProbe({ blockSize: 128, kernelLength: 65 });
  for (const Backend of [DeferredTailConvolutionBackend, MergedFinalBlockConvolutionBackend]) {
    const y = new Backend(probe.kernel, { blockSize: 128 }).processAll(probe.input);
    const beats = inspectTailBeats(y, probe.input, { groupDelaySamples: probe.groupDelaySamples });
    assert.equal(beats.pass, true, JSON.stringify(beats.extraPeaks));
    const diff = differenceToOriginalTrack(y, probe.input, probe.groupDelaySamples);
    assert.ok(diff.alignedMaxAbs < 1e-8);
  }

  const good = new DeferredTailConvolutionBackend(probe.kernel).processAll(probe.input);
  const bad = Float64Array.from([...good, ...good.subarray(good.length - 64)]);
  const ref = convolveDirect(probe.input, probe.kernel);
  assert.equal(convolutionErrorMetrics(bad, ref).lengthMatch, false);
  const duplicatedBeats = inspectTailBeats(bad, probe.input, { groupDelaySamples: probe.groupDelaySamples });
  // 追加段没有完整高峰时，长度证据仍必须失败；人为追加一个尾拍则 beat 检查必须失败。
  const withExtra = bad.slice();
  withExtra[good.length + 10] = 1;
  const beats2 = inspectTailBeats(withExtra, probe.input, { groupDelaySamples: probe.groupDelaySamples, threshold: 0.5 });
  assert.equal(beats2.pass, false);
  assert.equal(convolutionErrorMetrics(bad, ref).pass, false);
});

test('边录边换核：两种冲洗约定都等于分段直接卷积，冲激位置/相位一致', () => {
  const h1 = kernel(127);
  const h2 = kernel(255);
  const x = Float64Array.from({ length: 5000 }, (_, i) => Math.sin(i * 0.113) * 0.5);
  const at = 1877;
  const ref = convolveSwitchedDirect(x, [{ sample: 0, kernel: h1 }, { sample: at, kernel: h2 }]);

  const recorder = new HotSwapConvolutionRecorder(h1);
  recorder.push(x.subarray(0, at - 10));
  recorder.push(x.subarray(at - 10, at));
  const switchedAt = recorder.replaceKernel(h2);
  assert.equal(switchedAt, at);
  recorder.push(x.subarray(at));
  const result = recorder.finish();
  assert.ok(result.error.pass, `hot recorder err=${result.error.maxAbs}`);

  for (const convention of ['deferred-drain', 'merged-final']) {
    const rendered = renderSwitchedWithBackend(x, [
      { sample: 0, kernel: h1 }, { sample: at, kernel: h2 },
    ], { blockSize: 128, tailConvention: convention });
    const err = convolutionErrorMetrics(rendered.output, ref);
    assert.ok(err.pass, `${convention} max=${err.maxAbs}`);
  }

  const imp = new Float64Array(4096);
  const p = 2600;
  imp[p] = 1;
  const impRef = convolveSwitchedDirect(imp, [{ sample: 0, kernel: h1 }, { sample: at, kernel: h2 }]);
  const impOut = renderSwitchedWithBackend(imp, [{ sample: 0, kernel: h1 }, { sample: at, kernel: h2 }]).output;
  let peak = 0;
  for (let i = p; i < Math.min(impOut.length, p + h2.length); i++) if (Math.abs(impOut[i]) > Math.abs(impOut[peak])) peak = i;
  let refPeak = 0;
  for (let i = p; i < Math.min(impRef.length, p + h2.length); i++) if (Math.abs(impRef[i]) > Math.abs(impRef[refPeak])) refPeak = i;
  assert.equal(peak, refPeak);
  assert.equal(Math.sign(impOut[peak]), Math.sign(impRef[refPeak]));
});

test('Float32 后端使用明确更宽容差，Float64 容差不得伪报', () => {
  assert.ok(TOLERANCE_PROFILES.float32.maxAbs > TOLERANCE_PROFILES.float64.maxAbs);
  const ref = new Float64Array(1000);
  const actual = Float64Array.from(ref, () => 0);
  for (let i = 0; i < actual.length; i++) actual[i] = ref[i] + 1e-6;
  assert.equal(convolutionErrorMetrics(actual, ref, { tolerance: TOLERANCE_PROFILES.float32 }).pass, true);
  assert.equal(convolutionErrorMetrics(actual, ref, { tolerance: TOLERANCE_PROFILES.float64 }).pass, false);
});

test('旧 version=1 配置可迁移并真实处理长音频，迁移不裁切', () => {
  const old = {
    app: 'linear-phase-eq',
    version: 1,
    params: { points: points.map((p) => ({ ...p })), N: 333, firType: 'I', window: 'hann', sampleRate: 48000 },
    session: { scenario: 'old-tail', N1: 127, N2: 1023 },
  };
  const m = migrateFilterConfig(old);
  assert.equal(m.schemaVersion, 2);
  assert.equal(m.migration.migrated, true);
  assert.equal(m.params.N, 333);
  assert.equal(m.session.scenario, 'old-tail');
  assert.equal(m.migration.cropSamples, 0);

  const result = loadMigratedConfigAndProcess(old, undefined, { longAudioSamples: 60000 });
  assert.equal(result.ok, true, result.error && JSON.stringify({ max: result.error.maxAbs, pass: result.error.pass }));
  assert.equal(result.output.length, result.input.length + result.design.kernel.length - 1);
});

test('AudioWorklet 热换核：128 量子前缀与分段直接卷积在 Float32 容差内一致', () => {
  const here = path.dirname(fileURLToPath(import.meta.url));
  const workletSrc = readFileSync(path.join(here, '..', 'public', 'worklet', 'convolver-worklet.js'), 'utf8');
  let Captured = null;
  const sandbox = {
    AudioWorkletProcessor: class { constructor() { this.port = { onmessage: null, postMessage() {} }; } },
    registerProcessor: (_n, cls) => { Captured = cls; },
    sampleRate: 48000, console, Float32Array, Float64Array, Math,
  };
  vm.createContext(sandbox);
  vm.runInContext(workletSrc, sandbox);
  const proc = new Captured();
  const h1 = Float32Array.from(kernel(65));
  const h2 = Float32Array.from(kernel(129));
  const B = 128;
  const before = 8;
  const blocks = 32;
  const x = Float64Array.from({ length: B * blocks }, (_, i) => Math.sin(i * 0.091));
  const out = new Float64Array(x.length);
  proc.port.onmessage({ data: { type: 'kernel', kernel: h1 } });
  for (let b = 0; b < blocks; b++) {
    if (b === before) proc.port.onmessage({ data: { type: 'kernel', kernel: h2 } });
    const input = new Float32Array(B);
    input.set(x.subarray(b * B, (b + 1) * B));
    const outputs = [[new Float32Array(B)]];
    assert.equal(proc.process([[input]], outputs), true);
    out.set(outputs[0][0], b * B);
  }
  const ref = convolveSwitchedDirect(x, [{ sample: 0, kernel: h1 }, { sample: before * B, kernel: h2 }]);
  let maxAbs = 0;
  for (let i = 0; i < out.length; i++) maxAbs = Math.max(maxAbs, Math.abs(out[i] - ref[i]));
  assert.ok(maxAbs < TOLERANCE_PROFILES.float32.maxAbs, `hot swap err=${maxAbs}`);
});
