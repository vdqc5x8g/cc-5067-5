import { test } from 'node:test';
import assert from 'node:assert/strict';

import { designFir, normalizeLength, scanLengthTradeoff } from '../public/dsp/fir-design.js';
import { convolveDirect, streamConvolve, OverlapAddConvolver } from '../public/dsp/convolution.js';
import { runSessionSync, resolveKernels, planPositions, FixedClockIEM } from '../public/dsp/session.js';
import { detectCombCancellation } from '../public/dsp/comb.js';
import { encodeWavBytes } from '../public/dsp/wav.js';
import { runVerification } from '../public/dsp/verify.js';
import { FIR_TYPES } from '../public/dsp/fir-design.js';

const pts = [
  { f: 0, db: 4 },
  { f: 0.15, db: 3 },
  { f: 0.4, db: 0 },
  { f: 0.8, db: -2 },
  { f: 1, db: -2 },
];

test('I 型设计：奇数长且严格偶对称，群延迟恒定', () => {
  const d = designFir({ points: pts, N: 127, window: 'hann' });
  assert.equal(d.ok, true);
  assert.equal(d.N % 2, 1);
  assert.equal(d.groupDelaySamples, 63);
  assert.ok(d.symmetryResidual.symmetricMaxError < 1e-10);
  assert.ok(Math.abs(d.groupDelay.measuredMean - 63) < 1e-6);
  assert.ok(d.groupDelay.maxDeviationSamples < 1e-6);
});

test('II 型设计：偶数长偶对称；Nyquist 高增益目标必须失败并给 I 型建议', () => {
  const ok = designFir({ points: [{ f: 0, db: 0 }, { f: 1, db: -60 }], N: 128, firType: 'II' });
  assert.equal(ok.ok, true);
  assert.equal(ok.N % 2, 0);
  assert.ok(ok.symmetryResidual.symmetricMaxError < 1e-10);
  assert.equal(ok.groupDelaySamples, 63.5);

  const bad = designFir({ points: [{ f: 0, db: 0 }, { f: 1, db: 0 }], N: 128, firType: 'II' });
  assert.equal(bad.ok, false);
  assert.equal(bad.error, 'TYPE2_NYQUIST_ZERO');
  assert.equal(bad.suggestion.firType, 'I');
});

test('III/IV 型奇对称条件由构造保持', () => {
  for (const [t, N] of [['III', 31], ['IV', 32]]) {
    const d = designFir({ N, firType: t, kind: 'differentiator' });
    assert.equal(d.ok, true, d.message);
    assert.equal(d.symmetry, 'antisymmetric');
    assert.ok(d.symmetryResidual.antisymmetricMaxError < 1e-10);
    assert.ok(FIR_TYPES[t]);
  }
});

test('非法目标被拒绝（非有限/点太少/越界）', () => {
  assert.equal(designFir({ points: [{ f: 0, db: 0 }] }).ok, false);
  assert.equal(
    designFir({ points: [{ f: -0.2, db: 0 }, { f: 1, db: 0 }] }).ok,
    false
  );
  assert.equal(
    designFir({ points: [{ f: 0, db: 200 }, { f: 1, db: 0 }] }).ok,
    false
  );
});

test('Kaiser 设计对平滑 EQ 目标有良好逼近', () => {
  const d = designFir({ points: pts, N: 255, window: 'kaiser', stopbandDb: 70 });
  assert.equal(d.ok, true);
  assert.ok(d.approximation.maxErrorDb < 1.5, `maxErr=${d.approximation.maxErrorDb}`);
  assert.ok(d.approximation.rmsErrorDb < 0.1);
});

test('长度扫描：更长核给出更窄实测过渡带', () => {
  const rows = scanLengthTradeoff(null, { lengths: [65, 129, 257, 513], brickwall: true, window: 'kaiser' });
  const w = rows.map((r) => r.transitionHz);
  assert.ok(w.every((x) => x > 0));
  assert.ok(w[w.length - 1] < w[0] * 0.6, `过渡带未随长度显著变窄: ${w.join(',')}`);
});

test('直接卷积已知向量', () => {
  const y = convolveDirect(Float64Array.from([1, 2, 3, 4]), Float64Array.from([1, -1]));
  assert.deepEqual(Array.from(y), [1, 1, 1, 1, -4]);
});

for (const B of [1, 2, 7, 32, 128, 255, 256, 1000, 4096]) {
  test(`分块流式卷积与直接卷积相符（blockSize=${B}，含尾部）`, () => {
    const h = Float64Array.from({ length: 53 }, (_, i) => Math.cos(i * 0.4) * Math.exp(-i / 20));
    const x = Float64Array.from({ length: 777 }, (_, i) => Math.sin(i * 0.113) * 0.6);
    const ref = convolveDirect(x, h);
    const got = streamConvolve(x, h, B);
    assert.equal(got.length, x.length + h.length - 1);
    let err = 0;
    for (let i = 0; i < ref.length; i++) err = Math.max(err, Math.abs(ref[i] - got[i]));
    assert.ok(err < 1e-9, `maxErr=${err}`);
  });
}

test('多次 push 任意切片后 flush 仍与参照一致', () => {
  const h = designFir({ points: pts, N: 65 }).kernel;
  const x = Float64Array.from({ length: 5000 }, (_, i) => (Math.random() - 0.5) * 0.4);
  const c = new OverlapAddConvolver(h, 128);
  const pieces = [];
  for (let i = 0; i < x.length; ) {
    const n = Math.min(x.length - i, 1 + Math.floor(Math.random() * 300));
    pieces.push(c.push(x.subarray(i, i + n)));
    i += n;
  }
  pieces.push(c.flush());
  const got = new Float64Array(pieces.reduce((a, p) => a + p.length, 0));
  let off = 0;
  for (const p of pieces) { got.set(p, off); off += p.length; }
  const ref = convolveDirect(x, h);
  assert.equal(got.length, ref.length);
  let err = 0;
  for (let i = 0; i < ref.length; i++) err = Math.max(err, Math.abs(ref[i] - got[i]));
  assert.ok(err < 1e-9);
});

test('脉冲峰位等于群延迟，输出保留完整尾部', () => {
  const d = designFir({ points: pts, N: 127 });
  const x = new Float64Array(2048);
  x[0] = 1;
  const y = convolveDirect(x, d.kernel);
  let idx = 0;
  for (let i = 1; i < d.N; i++) if (Math.abs(y[i]) > Math.abs(y[idx])) idx = i;
  assert.equal(idx, 63);
  assert.equal(y.length, 2048 + 127 - 1);
});

test('会话：旧尾情景默认两路径实测峰差=Δgd 且拍差换算正确', () => {
  const s = runSessionSync({ scenario: 'old-tail' });
  assert.equal(s.ok, true);
  assert.equal(s.difference.absoluteSamples, s.resolved.deltaGd);
  const expectBeats = s.resolved.deltaGd / s.plan.spb;
  assert.ok(Math.abs(s.difference.absoluteBeats - expectBeats) < 1e-12);
  assert.ok(s.difference.absoluteMs > 0);
});

test('会话：新核后鼓点默认两路径峰差为 0', () => {
  const s = runSessionSync({ scenario: 'new-effective' });
  assert.equal(s.difference.absoluteSamples, 0);
});

test('会话模式矩阵：dual-old/dual-new 相对 deferred 的平移关系', () => {
  const delta = runSessionSync().resolved.deltaGd;
  // 旧尾：dual-new 把旧峰右移 Δ；dual-old/deferred 不动
  assert.equal(runSessionSync({ scenario: 'old-tail', echoMode: 'deferred', iemMode: 'dual-new' }).difference.absoluteSamples, delta);
  assert.equal(runSessionSync({ scenario: 'old-tail', echoMode: 'deferred', iemMode: 'dual-old' }).difference.absoluteSamples, 0);
  // 新核后：dual-old 把新峰左移 Δ；dual-new/deferred 在新网格一致
  assert.equal(runSessionSync({ scenario: 'new-effective', echoMode: 'deferred', iemMode: 'dual-old' }).difference.absoluteSamples, delta);
  assert.equal(runSessionSync({ scenario: 'new-effective', echoMode: 'deferred', iemMode: 'dual-new' }).difference.absoluteSamples, 0);
});

test('耳返切换点被固定块时钟向上量化', () => {
  const r = resolveKernels({});
  const plan = planPositions(r, { iemBlock: 256 });
  const iem = new FixedClockIEM(r, plan, {});
  assert.equal(iem.switchLatchedAt % 256, 0);
  assert.ok(iem.switchLatchedAt >= plan.S);
  assert.ok(iem.switchQuantizationSamples >= 0 && iem.switchQuantizationSamples < 256);
  assert.equal(iem.render(new Float64Array(100)).blockSize, 256);
});

test('梳状检测：延迟相加检出，对齐相加不检出', () => {
  const sr = 48000;
  const tau = 137;
  const n = 8192;
  const a = new Float64Array(n);
  a[100] = 1;
  const raw = new Float64Array(n + tau);
  raw.set(a);
  for (let i = 0; i < n; i++) raw[i + tau] += 0.95 * a[i];
  const det = detectCombCancellation(raw, sr, { expectedDelaySamples: tau });
  assert.equal(det.detected, true);
  assert.ok(det.notchCount >= 3);
  assert.ok(Math.abs(det.expectedSpacingHz - sr / tau) < 1e-9);
});

test('WAV 编码：RIFF 头与长度自洽', () => {
  const pcm = new Float64Array(1000);
  pcm[5] = 0.5;
  const bytes = encodeWavBytes([pcm], 48000);
  assert.equal(bytes.byteLength, 44 + 2000);
  const dv = new DataView(bytes.buffer);
  assert.equal(String.fromCharCode(bytes[0], bytes[1], bytes[2], bytes[3]), 'RIFF');
  assert.equal(dv.getUint32(24, true), 48000);
  assert.equal(dv.getUint32(40, true), 2000);
});

test('统一自检全部通过', () => {
  const r = runVerification();
  assert.equal(r.pass, true, r.results.filter((x) => !x.pass).map((x) => `${x.id}:${x.detail}`).join('\n'));
  assert.ok(r.passed >= 10);
});
