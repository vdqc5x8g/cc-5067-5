// 两条交付路径的样本精确会话模型：
//   1) MasterEchoBus  —— 可等候的“母带回印”，离线逐样本渲染，切换点样本精确；
//   2) FixedClockIEM —— 固定块时钟的鼓手耳返，核切换只在块边界锁存。
//
// 交接策略：
//   deferred      旧核继续负责 S 之前的全部输入（含完整尾部），新核负责 S 起的新输入；
//   dual-old      双核算全曲，按群延迟对齐到旧核时标后等功率交叉淡化（对齐旧尾声）；
//   dual-new      双核算全曲，对齐到新核时标（旧结果额外平移 Δgd，切换新时标）。
//
// 所有峰位都从真实渲染缓冲里用 findPeak 量出，UI 移动显示标尺不会改变这些数字。

import { designFir, normalizeLength } from './fir-design.js';
import { processWithFirBackend, CONVOLUTION_BACKEND_VERSION } from './fir-backend.js';
import { findPeak, beatSamples, sampleToBeat, generateImpulse, placeDrum } from './audio.js';
import { detectCombCancellation } from './comb.js';
import { defaultTargetPoints } from './target.js';

export const SESSION_DEFAULTS = {
  sampleRate: 48000,
  bpm: 124,
  N1: 127,
  N2: 1023,
  window: 'hann',
  iemBlock: 256,
  handoff: { echo: 'deferred', iem: 'dual-new' },
  scenario: 'old-tail',
  material: 'click',
  fadeSamples: 512,
};

export function resolveKernels(config = {}) {
  const sampleRate = config.sampleRate || SESSION_DEFAULTS.sampleRate;
  const points = config.targetPoints || defaultTargetPoints();
  const window = config.window || SESSION_DEFAULTS.window;
  const N1 = normalizeLength(config.N1 || SESSION_DEFAULTS.N1, 'I');
  const N2 = normalizeLength(config.N2 || SESSION_DEFAULTS.N2, 'I');
  const d1 = designFir({ points, N: N1, firType: 'I', window, sampleRate });
  const d2 = designFir({ points, N: N2, firType: 'I', window, sampleRate });
  if (!d1.ok || !d2.ok) {
    return { ok: false, message: d1.message || d2.message, failed: !d1.ok ? d1 : d2 };
  }
  const h1 = Float64Array.from(d1.kernel);
  const h2 = Float64Array.from(d2.kernel);
  const gd1 = d1.groupDelaySamples;
  const gd2 = d2.groupDelaySamples;
  return { ok: true, h1, h2, d1, d2, N1, N2, gd1, gd2, deltaGd: gd2 - gd1, sampleRate };
}

// 规划切换点与孤立鼓点位置。
export function planPositions(resolved, config = {}) {
  const { sampleRate, gd1 } = resolved;
  const bpm = config.bpm || SESSION_DEFAULTS.bpm;
  const spb = beatSamples(bpm, sampleRate);
  const S = config.switchSample != null ? Math.round(config.switchSample) : 2 * spb;
  const B = config.iemBlock || SESSION_DEFAULTS.iemBlock;
  const SIem = Math.ceil(S / B) * B;
  const F = config.fadeSamples || SESSION_DEFAULTS.fadeSamples;
  const scenario = config.scenario || SESSION_DEFAULTS.scenario;

  // old-tail: 鼓的输入点在 S 之前 gd1+8，旧核声学峰在 S-8，
  // 而旧核尾部一直延续到 S+gd1-8 > S —— 增大核长时鼓点仍在旧尾里。
  const pOld = S - gd1 - 8;
  // new-effective: 鼓点放在切换 + 整块时钟量化 + 淡化区之后，确保由新核完整处理。
  const pNew = S + F + B + 256;
  const hitPosition = scenario === 'new-effective' ? pNew : pOld;

  return { spb, bpm, S, SIem, B, F, scenario, hitPosition };
}

export function buildScenarioInput(resolved, plan, config = {}) {
  const { sampleRate } = resolved;
  const material = config.material || SESSION_DEFAULTS.material;
  const p = plan.hitPosition;
  // 尾部留出 N2 长度 + 余量，保证任何渲染方式都不截尾
  const totalLength = p + resolved.N2 + Math.floor(0.2 * sampleRate);
  if (material === 'kick') {
    return placeDrum(p, sampleRate, totalLength, { amplitude: 0.95 });
  }
  // click：单位脉冲式“鼓点击发”，时间基准最严格
  return generateImpulse({ length: totalLength, position: p, amplitude: 1 });
}

function crossfadeGains(t, t0, F) {
  let u = (t - t0) / F;
  if (u <= 0) return { g1: 1, g2: 0 };
  if (u >= 1) return { g1: 0, g2: 1 };
  const theta = u * Math.PI * 0.5;
  return { g1: Math.cos(theta), g2: Math.sin(theta) };
}

function firThroughNewBackend(x, h) {
  return processWithFirBackend(x, h, {
    blockSize: 128,
    tailConvention: 'deferred-drain',
    precision: 'float64',
  }).output;
}

// 单条路径渲染。switchAt 为本路径实际生效的切换样本（耳返会被块量化）。
export function renderPath(x, resolved, plan, options = {}) {
  const { h1, h2, gd1, gd2, deltaGd } = resolved;
  const mode = options.mode || 'deferred'; // deferred | dual-old | dual-new
  const switchAt = options.switchAt != null ? options.switchAt : plan.S;
  const PREROLL = gd2; // 两条路径共享同一文件时间原点
  const T = x.length;
  const outLen = PREROLL + T + resolved.N2 + deltaGd + 32;
  const y = new Float64Array(outLen);

  if (mode === 'deferred') {
    const a = firThroughNewBackend(x.subarray(0, switchAt), h1);
    for (let i = 0; i < a.length; i++) y[PREROLL + i] += a[i];
    const b = firThroughNewBackend(x.subarray(switchAt), h2);
    const off = PREROLL + switchAt;
    for (let i = 0; i < b.length; i++) y[off + i] += b[i];
  } else if (mode === 'dual-old' || mode === 'dual-new') {
    const c1 = firThroughNewBackend(x, h1);
    const c2 = firThroughNewBackend(x, h2);
    let off1, off2, t0;
    if (mode === 'dual-old') {
      // c2 左移 Δ，与 c1 对齐到旧群延迟网格
      off1 = PREROLL;
      off2 = PREROLL - deltaGd;
      t0 = switchAt + gd1;
    } else {
      // c1 右移 Δ，与 c2 对齐到新群延迟网格（切换新时标）
      off1 = PREROLL + deltaGd;
      off2 = PREROLL;
      t0 = switchAt + gd2;
    }
    const F = plan.F;
    for (let i = 0; i < y.length; i++) {
      const t = i - PREROLL;
      const { g1, g2 } = crossfadeGains(t, t0, F);
      let v = 0;
      const j1 = i - off1;
      if (g1 > 0 && j1 >= 0 && j1 < c1.length) v += g1 * c1[j1];
      const j2 = i - off2;
      if (g2 > 0 && j2 >= 0 && j2 < c2.length) v += g2 * c2[j2];
      y[i] = v;
    }
  } else {
    throw new Error('未知交接模式: ' + mode);
  }

  return { audio: y, preroll: PREROLL, mode, switchAt };
}

// 从渲染缓冲量峰，并换算到输入时钟与拍坐标。
export function measurePath(rendered, resolved, plan) {
  const { audio, preroll, mode, switchAt } = rendered;
  const sr = resolved.sampleRate;
  const peak = findPeak(audio, 0, audio.length - 1);
  const clockSample = peak.index - preroll;
  return {
    mode,
    switchAt,
    peakSampleFile: peak.index,
    peakSubSampleFile: peak.subSampleIndex,
    peakSampleClock: clockSample,
    peakTimeSec: clockSample / sr,
    peakBeat: clockSample / plan.spb,
    peakAmplitude: peak.absAmplitude,
    prerollSamples: preroll,
    durationSamples: audio.length,
    durationSec: audio.length / sr,
  };
}

// 可等候的母带回印总线：send() 返回 Promise，resolve 出完整回印缓冲与测量。
export class MasterEchoBus {
  constructor(resolved, plan, config = {}) {
    this.resolved = resolved;
    this.plan = plan;
    this.mode = config.echoMode || (SESSION_DEFAULTS.handoff.echo);
  }

  async send(x) {
    const t0 = Date.now();
    // 让出事件循环一次，使“可等候”语义真实（UI 会先显示等待状态）
    await new Promise((r) => setTimeout(r, 20));
    const rendered = renderPath(x, this.resolved, this.plan, {
      mode: this.mode,
      switchAt: this.plan.S,
    });
    const measurement = measurePath(rendered, this.resolved, this.plan);
    this.lastReturn = rendered.audio;
    return {
      ...measurement,
      awaitMs: Date.now() - t0,
      status: 'returned',
      audio: rendered.audio,
    };
  }
}

// 固定块时钟鼓手耳返：切换点向上量化到块边界，输出按 B 样本块交付。
export class FixedClockIEM {
  constructor(resolved, plan, config = {}) {
    this.resolved = resolved;
    this.plan = plan;
    this.blockSize = plan.B;
    this.mode = config.iemMode || SESSION_DEFAULTS.handoff.iem;
    this.clockSample = 0;
    this.switchLatchedAt = plan.SIem;
    this.switchQuantizationSamples = plan.SIem - plan.S;
  }

  render(x) {
    const rendered = renderPath(x, this.resolved, this.plan, {
      mode: this.mode,
      switchAt: this.plan.SIem,
    });
    // 块时钟交付：尾部补零到整块
    const B = this.blockSize;
    const paddedLen = Math.ceil(rendered.audio.length / B) * B;
    const padded = new Float64Array(paddedLen);
    padded.set(rendered.audio);
    const blocks = paddedLen / B;
    this.clockSample = paddedLen;
    const measurement = measurePath(
      { ...rendered, audio: padded },
      this.resolved,
      this.plan
    );
    return {
      ...measurement,
      audio: padded,
      rawAudio: rendered.audio,
      blockSize: B,
      blocks,
      clockSamples: paddedLen,
      clockSec: paddedLen / this.resolved.sampleRate,
      switchLatchedAt: this.plan.SIem,
      switchQuantizationSamples: this.plan.SIem - this.plan.S,
    };
  }
}

// 梳状自检信号：同一脉冲分别过 h1/h2 后，“未对齐直接相加”与“按 gd 对齐相加”。
export function combSelfCheck(resolved) {
  const { h1, h2, gd1, gd2, deltaGd, sampleRate } = resolved;
  const L = Math.max(h1.length, h2.length) + deltaGd + 64;
  const raw = new Float64Array(L);
  for (let i = 0; i < h1.length; i++) raw[i] += h1[i];
  for (let i = 0; i < h2.length; i++) raw[i] += h2[i];

  const aligned = new Float64Array(L);
  // 对齐到新网格：h2 在 0，h1 右移 Δ
  for (let i = 0; i < h2.length; i++) aligned[i] += h2[i];
  for (let i = 0; i < h1.length; i++) aligned[i + deltaGd] += h1[i];

  return {
    delaySamples: deltaGd,
    raw: detectCombCancellation(raw, sampleRate, { expectedDelaySamples: deltaGd }),
    aligned: detectCombCancellation(aligned, sampleRate, { expectedDelaySamples: deltaGd }),
  };
}

// 高层：跑完整双路径会话，返回 {echo, iem, difference, comb, ...} 与音频缓冲。
export async function runSession(config = {}) {
  const resolved = resolveKernels(config);
  if (!resolved.ok) return { ok: false, error: resolved.message };
  const plan = planPositions(resolved, config);
  const x = buildScenarioInput(resolved, plan, config);

  const echoBus = new MasterEchoBus(resolved, plan, { echoMode: config.echoMode });
  const iemBus = new FixedClockIEM(resolved, plan, { iemMode: config.iemMode });

  const echo = await echoBus.send(x);
  const iem = iemBus.render(x);

  const diffSamples = Math.abs(echo.peakSampleFile - iem.peakSampleFile);
  const spb = plan.spb;
  const difference = {
    echoPeakFile: echo.peakSampleFile,
    iemPeakFile: iem.peakSampleFile,
    signedSamples: iem.peakSampleFile - echo.peakSampleFile,
    absoluteSamples: diffSamples,
    absoluteMs: (diffSamples / resolved.sampleRate) * 1000,
    absoluteBeats: diffSamples / spb,
    beatEcho: echo.peakBeat,
    beatIem: iem.peakBeat,
    absoluteBeatDifference: Math.abs(echo.peakBeat - iem.peakBeat),
  };

  const comb = combSelfCheck(resolved);

  const info = {
    ok: true,
    convolutionBackendVersion: CONVOLUTION_BACKEND_VERSION,
    backend: {
      version: CONVOLUTION_BACKEND_VERSION,
      renderer: 'processWithFirBackend/FirBackendAdapter',
      tailConvention: 'deferred-drain',
    },
    config: {
      sampleRate: resolved.sampleRate,
      bpm: config.bpm || SESSION_DEFAULTS.bpm,
      N1: resolved.N1,
      N2: resolved.N2,
      window: config.window || SESSION_DEFAULTS.window,
      scenario: plan.scenario,
      material: config.material || SESSION_DEFAULTS.material,
      echoMode: echoBus.mode,
      iemMode: iemBus.mode,
      iemBlock: plan.B,
    },
    kernel: {
      N1: resolved.N1,
      N2: resolved.N2,
      gd1: resolved.gd1,
      gd2: resolved.gd2,
      deltaGd: resolved.deltaGd,
      latency1Ms: (resolved.gd1 / resolved.sampleRate) * 1000,
      latency2Ms: (resolved.gd2 / resolved.sampleRate) * 1000,
    },
    positions: {
      hitInputSample: plan.hitPosition,
      switchCommandSample: plan.S,
      switchIemSample: plan.SIem,
      switchQuantizationSamples: plan.SIem - plan.S,
      spb,
      hitBeat: plan.hitPosition / spb,
      oldTailEndAtSwitch: plan.S + resolved.gd1 - 8,
      fadeSamples: plan.F,
    },
    echo: sanitizeMeasurement(echo),
    iem: sanitizeMeasurement(iem),
    difference,
    comb,
  };

  return {
    ...info,
    _audio: { echo: echo.audio, iem: iem.audio, input: x },
  };
}

function sanitizeMeasurement(m) {
  const { audio, ...rest } = m;
  void audio;
  return rest;
}

// 同步版本（测试/API 不需要 await 时序时使用）
export function runSessionSync(config = {}) {
  const resolved = resolveKernels(config);
  if (!resolved.ok) return { ok: false, error: resolved.message };
  const plan = planPositions(resolved, config);
  const x = buildScenarioInput(resolved, plan, config);
  const echoRendered = renderPath(x, resolved, plan, {
    mode: config.echoMode || SESSION_DEFAULTS.handoff.echo,
    switchAt: plan.S,
  });
  const iem = new FixedClockIEM(resolved, plan, { iemMode: config.iemMode }).render(x);
  const echoMeasure = measurePath(echoRendered, resolved, plan);
  const diffSamples = Math.abs(echoMeasure.peakSampleFile - iem.peakSampleFile);
  return {
    ok: true,
    convolutionBackendVersion: CONVOLUTION_BACKEND_VERSION,
    backend: {
      version: CONVOLUTION_BACKEND_VERSION,
      renderer: 'processWithFirBackend/FirBackendAdapter',
      tailConvention: 'deferred-drain',
    },
    resolved,
    plan,
    input: x,
    echoAudio: echoRendered.audio,
    iemAudio: iem.audio,
    echoMeasure,
    iemMeasure: iem,
    difference: {
      absoluteSamples: diffSamples,
      absoluteMs: (diffSamples / resolved.sampleRate) * 1000,
      absoluteBeats: diffSamples / plan.spb,
      signedSamples: iem.peakSampleFile - echoMeasure.peakSampleFile,
    },
    comb: combSelfCheck(resolved),
  };
}

export { sampleToBeat };
