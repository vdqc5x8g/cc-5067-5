// 测试素材与客观测量：脉冲、对数扫频、尖锐瞬态鼓点、峰位检测、拍换算。

export function beatSamples(bpm, sampleRate) {
  return Math.round((60 / bpm) * sampleRate);
}

export function generateImpulse({ sampleRate = 48000, length = 4800, position = 0, amplitude = 1 } = {}) {
  const x = new Float64Array(length);
  if (position >= 0 && position < length) x[position] = amplitude;
  return x;
}

// 对数扫频 20Hz -> 20kHz（不超过 Nyquist*0.9），带 5ms 淡入淡出避免端点咔哒。
export function generateSweep({ sampleRate = 48000, durationSec = 3, f0 = 20, f1, amplitude = 0.8 } = {}) {
  const n = Math.floor(durationSec * sampleRate);
  const x = new Float64Array(n);
  const endF = Math.min(f1 || 20000, sampleRate * 0.45);
  const k = endF / f0;
  for (let i = 0; i < n; i++) {
    const t = i / sampleRate;
    const phase = 2 * Math.PI * f0 * durationSec * (Math.pow(k, t / durationSec) - 1) / Math.log(k);
    let env = amplitude;
    const fade = Math.floor(0.005 * sampleRate);
    if (i < fade) env *= i / fade;
    if (i > n - fade) env *= (n - i) / fade;
    x[i] = env * Math.sin(phase);
  }
  return x;
}

// 单声尖锐底鼓：0 时刻为全幅脉冲式攻击（保证 argmax 在第 0 样本），
// 后接快速下滑正弦与指数衰减作为“鼓身”，整体仍有尖锐瞬态。
export function drumHit(length, sampleRate, { amplitude = 0.95 } = {}) {
  const x = new Float64Array(length);
  for (let i = 0; i < length; i++) {
    const t = i / sampleRate;
    const f = 150 * Math.exp(-t * 28) + 48;
    const body = 0.32 * Math.exp(-t * 9) * Math.sin(2 * Math.PI * f * t);
    const click = i === 0 ? amplitude : 0;
    x[i] = Math.max(-1, Math.min(1, click + body));
  }
  // 强制第 0 样本为绝对峰，保证瞬态时间基准无歧义
  x[0] = amplitude;
  return x;
}

// 把单声鼓放到 position。totalLength 可选。
export function placeDrum(position, sampleRate, totalLength, opts = {}) {
  const hitLen = opts.hitLength || Math.floor(0.4 * sampleRate);
  const total = totalLength || position + hitLen + 1;
  const x = new Float64Array(total);
  const hit = drumHit(hitLen, sampleRate, opts);
  x.set(hit.subarray(0, Math.min(hitLen, total - position)), position);
  return x;
}

// 简单鼓组片段：按拍放置底鼓 + 噪声军鼓/拍手，用于实时试听。
export function generateDrumLoop({ sampleRate = 48000, bpm = 124, bars = 2, amplitude = 0.9 } = {}) {
  const spb = beatSamples(bpm, sampleRate);
  const total = spb * 4 * bars + Math.floor(0.5 * sampleRate);
  const x = new Float64Array(total);
  for (let bar = 0; bar < bars; bar++) {
    for (let b = 0; b < 4; b++) {
      const pos = (bar * 4 + b) * spb;
      const hit = drumHit(Math.floor(0.35 * sampleRate), sampleRate, { amplitude: amplitude * (b === 0 ? 1 : 0.85) });
      for (let i = 0; i < hit.length && pos + i < total; i++) x[pos + i] += hit[i];
      // 反拍噪声拍手
      if (b === 1 || b === 3) {
        const nzLen = Math.floor(0.12 * sampleRate);
        for (let i = 0; i < nzLen && pos + i < total; i++) {
          const env = Math.exp(-i / (sampleRate * 0.03));
          x[pos + i] += amplitude * 0.25 * env * (Math.random() * 2 - 1);
        }
      }
    }
  }
  const peak = x.reduce((m, v) => Math.max(m, Math.abs(v)), 0);
  if (peak > 1) for (let i = 0; i < x.length; i++) x[i] /= peak;
  return x;
}

// 在 [from,to]（含边界，可越界自动裁剪）内找绝对幅度峰。
// 返回整数样本峰与抛物插值的亚样本估计。
export function findPeak(signal, from = 0, to = signal.length - 1) {
  const lo = Math.max(0, Math.floor(from));
  const hi = Math.min(signal.length - 1, Math.ceil(to));
  if (hi < lo) return { index: -1, amplitude: 0 };
  let idx = lo;
  let amp = Math.abs(signal[lo]);
  for (let i = lo + 1; i <= hi; i++) {
    const a = Math.abs(signal[i]);
    if (a > amp) { amp = a; idx = i; }
  }
  // 抛物插值（用带符号样本）
  let sub = idx;
  if (idx > 0 && idx < signal.length - 1) {
    const y0 = signal[idx - 1];
    const y1 = signal[idx];
    const y2 = signal[idx + 1];
    const denom = y0 - 2 * y1 + y2;
    if (Math.abs(denom) > 1e-12) {
      const delta = 0.5 * (y0 - y2) / denom;
      if (Number.isFinite(delta) && Math.abs(delta) <= 1) sub = idx + delta;
    }
  }
  return { index: idx, subSampleIndex: sub, amplitude: signal[idx], absAmplitude: amp };
}

export function sampleToBeat(sample, bpm, sampleRate) {
  const spb = beatSamples(bpm, sampleRate);
  return sample / spb;
}

// 前振铃：对称核主峰之前的能量占比与相对峰值电平。
export function preRingingMetrics(kernel, gd) {
  const center = Math.round(gd != null ? gd : (kernel.length - 1) / 2);
  let preE = 0;
  let postE = 0;
  let totalE = 0;
  let prePeak = 0;
  let mainPeak = 0;
  for (let i = 0; i < kernel.length; i++) {
    const e = kernel[i] * kernel[i];
    totalE += e;
    if (i < center) preE += e;
    else postE += e;
    if (i === center) mainPeak = Math.max(mainPeak, Math.abs(kernel[i]));
    if (i < center) prePeak = Math.max(prePeak, Math.abs(kernel[i]));
    else mainPeak = Math.max(mainPeak, Math.abs(kernel[i]));
  }
  return {
    centerSample: center,
    preEnergyRatio: preE / totalE,
    preEnergyDb: 10 * Math.log10((preE || 1e-20) / totalE),
    preToPeakDb: 20 * Math.log10((prePeak || 1e-12) / (mainPeak || 1)),
  };
}
