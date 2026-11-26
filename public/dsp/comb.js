// 梳状抵消检测：两个相差 τ 样本的近似同幅响应相加时，
// 频谱出现间隔 ≈ fs/τ 的周期性深零点。这里用 FFT 包络陷波搜索 +
// 间隔规律性来判定，而不是只看时域相关性。

import { frequencyResponse } from './response.js';

function median(arr) {
  if (!arr.length) return null;
  const a = arr.slice().sort((x, y) => x - y);
  return a[Math.floor(a.length / 2)];
}

export function detectCombCancellation(signal, sampleRate, opts = {}) {
  const nfft = opts.nfft || 16384;
  const depthThresholdDb = opts.depthThresholdDb ?? 9;
  const minNotches = opts.minNotches ?? 3;
  const fMin = opts.fMin ?? 50;
  const fMax = opts.fMax ?? Math.min(14000, sampleRate * 0.42);
  const expectedDelaySamples = opts.expectedDelaySamples ?? null;

  const resp = frequencyResponse(signal, sampleRate, nfft);
  const df = sampleRate / resp.nfft;
  const kMin = Math.max(2, Math.floor(fMin / df));
  const kMax = Math.min(resp.bins - 2, Math.ceil(fMax / df));

  // 滑动最大包络，窗宽取预期间隔的 ~0.6，否则按 400Hz 估计
  let envBins;
  if (expectedDelaySamples && expectedDelaySamples > 0) {
    const spacingHz = sampleRate / expectedDelaySamples;
    envBins = Math.max(5, Math.round((0.6 * spacingHz) / df));
  } else {
    envBins = Math.max(5, Math.round(400 / df));
  }
  const half = Math.floor(envBins / 2);

  const envelope = new Float64Array(resp.bins);
  for (let k = kMin; k <= kMax; k++) {
    let mx = -Infinity;
    for (let j = Math.max(0, k - half); j <= Math.min(resp.bins - 1, k + half); j++) {
      if (resp.magDb[j] > mx) mx = resp.magDb[j];
    }
    envelope[k] = mx;
  }

  // 候选陷波：局部极小且相对包络足够深
  const candidates = [];
  for (let k = kMin + 1; k < kMax; k++) {
    const db = resp.magDb[k];
    if (db > resp.magDb[k - 1] || db > resp.magDb[k + 1]) continue;
    const depth = envelope[k] - db;
    if (depth >= depthThresholdDb) {
      candidates.push({ bin: k, freq: k * df, depthDb: depth });
    }
  }

  // 非极大抑制：相邻 envBins/2 内只保留最深
  candidates.sort((a, b) => b.depthDb - a.depthDb);
  const picked = [];
  const minBinDist = Math.max(3, Math.floor(envBins / 3));
  for (const c of candidates) {
    if (picked.every((p) => Math.abs(p.bin - c.bin) >= minBinDist)) picked.push(c);
  }
  picked.sort((a, b) => a.bin - b.bin);

  const spacings = [];
  for (let i = 1; i < picked.length; i++) spacings.push(picked[i].freq - picked[i - 1].freq);
  const medianSpacing = median(spacings);
  let regularity = null;
  if (spacings.length >= 2 && medianSpacing > 0) {
    const dev = median(spacings.map((s) => Math.abs(s - medianSpacing)));
    regularity = dev / medianSpacing;
  }
  const deepest = picked.reduce((m, c) => Math.max(m, c.depthDb), 0);
  const medianDepth = median(picked.map((c) => c.depthDb));

  const expectedSpacingHz =
    expectedDelaySamples && expectedDelaySamples > 0 ? sampleRate / expectedDelaySamples : null;

  // 与预期间隔吻合度（若给出 τ）
  let spacingMatch = null;
  if (expectedSpacingHz && medianSpacing) {
    spacingMatch = Math.abs(medianSpacing - expectedSpacingHz) / expectedSpacingHz;
  }

  const enough = picked.length >= minNotches;
  const regular = regularity === null ? picked.length >= minNotches : regularity < 0.4;
  const deep = (medianDepth || 0) >= depthThresholdDb + 2 || deepest >= 18;
  const matchesExpected = spacingMatch === null || spacingMatch < 0.25;

  const detected = enough && regular && deep && matchesExpected;

  return {
    detected,
    notchCount: picked.length,
    notches: picked.slice(0, 24).map((c) => ({ freqHz: c.freq, depthDb: c.depthDb })),
    deepestDepthDb: deepest || null,
    medianDepthDb: medianDepth,
    medianSpacingHz: medianSpacing,
    expectedSpacingHz,
    spacingErrorRatio: spacingMatch,
    regularity,
    thresholdDb: depthThresholdDb,
  };
}
