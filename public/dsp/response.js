// 由实际内核计算“真正实现出来的”幅度、相位、群延迟与各种可检查指标。
// 界面上所有曲线都必须来自这里，而不是只画理想目标。

import { realFft } from './fft.js';
import { sampleTargetDb, sampleTargetMag, steepestSegment } from './target.js';

export function symmetryResidual(h) {
  const N = h.length;
  let sym = 0;
  let anti = 0;
  let energy = 0;
  for (let n = 0; n < N; n++) {
    energy = Math.max(energy, Math.abs(h[n]));
    sym = Math.max(sym, Math.abs(h[n] - h[N - 1 - n]));
    anti = Math.max(anti, Math.abs(h[n] + h[N - 1 - n]));
  }
  const scale = energy || 1;
  return {
    symmetricMaxError: sym / scale,
    antisymmetricMaxError: anti / scale,
    symmetricDb: 20 * Math.log10((sym || 1e-15) / scale),
  };
}

// FFT 频响。nfft 默认 >= 8N 且为 2 的幂。
export function frequencyResponse(kernel, sampleRate, nfft) {
  const N = kernel.length;
  let L = nfft;
  if (!L) {
    L = 1024;
    while (L < N * 8) L <<= 1;
    L = Math.min(L, 32768);
  }
  const H = realFft(kernel, L);
  // Q = FFT(n*h[n])，群延迟 gd = Re(Q / H)
  const weighted = new Float64Array(N);
  for (let n = 0; n < N; n++) weighted[n] = n * kernel[n];
  const Q = realFft(weighted, L);

  const bins = L / 2 + 1;
  const freqs = new Float64Array(bins);
  const mag = new Float64Array(bins);
  const magDb = new Float64Array(bins);
  const phase = new Float64Array(bins);
  const phaseUnwrapped = new Float64Array(bins);
  const gd = new Float64Array(bins);
  const gdValid = new Uint8Array(bins);

  let lastPhase = 0;
  for (let k = 0; k < bins; k++) {
    freqs[k] = (k * sampleRate) / L;
    const re = H.re[k];
    const im = H.im[k];
    const m = Math.hypot(re, im);
    mag[k] = m;
    magDb[k] = 20 * Math.log10(Math.max(m, 1e-8));
    let ph = Math.atan2(im, re);
    phase[k] = ph;
    if (k === 0) {
      phaseUnwrapped[k] = ph;
    } else {
      let prev = phase[k - 1];
      let d = ph - prev;
      while (d > Math.PI) { d -= 2 * Math.PI; ph -= 2 * Math.PI; }
      while (d < -Math.PI) { d += 2 * Math.PI; ph += 2 * Math.PI; }
      phaseUnwrapped[k] = phaseUnwrapped[k - 1] + d;
    }
    lastPhase = ph;

    const denom = re * re + im * im;
    if (denom > 1e-12) {
      // Re(Q * conj(H) / |H|^2)
      const qh = (Q.re[k] * re + Q.im[k] * im) / denom;
      gd[k] = qh;
      if (m > 1e-4 && Number.isFinite(qh) && Math.abs(qh) < N * 2) gdValid[k] = 1;
    }
  }
  void lastPhase;

  return { nfft: L, bins, freqs, mag, magDb, phase, phaseUnwrapped, gd, gdValid };
}

// 目标逼近误差（整个 0..Nyquist 网格）。
export function approximationError(resp, targetPoints, sampleRate) {
  const { bins, freqs, mag } = resp;
  let maxDb = -Infinity;
  let sumSq = 0;
  let count = 0;
  const errorCurve = new Float64Array(bins);
  for (let k = 0; k < bins; k++) {
    const f = freqs[k] / (sampleRate / 2);
    const targetDb = sampleTargetDb(targetPoints, Math.min(1, Math.max(0, f)));
    const actualDb = 20 * Math.log10(Math.max(mag[k], 1e-8));
    const e = actualDb - targetDb;
    errorCurve[k] = e;
    // 忽略目标极低且实际被数值底噪限制的区域，避免误差指标被 -160dB 假数据污染
    if (targetDb > -55) {
      maxDb = Math.max(maxDb, Math.abs(e));
      sumSq += e * e;
      count++;
    }
  }
  return {
    maxErrorDb: Number.isFinite(maxDb) ? maxDb : null,
    rmsErrorDb: count ? Math.sqrt(sumSq / count) : null,
    errorCurve,
  };
}

// 围绕目标最陡段，用实际响应的“目标变化量 10%↔90%”宽度估计过渡带。
export function measureTransitionBand(resp, targetPoints, sampleRate) {
  const seg = steepestSegment(targetPoints);
  if (!seg || Math.abs(seg.b.db - seg.a.db) < 6) {
    return { available: false, reason: '目标没有幅度变化 ≥6dB 的明显过渡段' };
  }
  const lowDb = seg.a.db;
  const highDb = seg.b.db;
  const dir = Math.sign(highDb - lowDb);
  const span = Math.abs(highDb - lowDb);
  const level10 = lowDb + (highDb - lowDb) * 0.1;
  const level90 = lowDb + (highDb - lowDb) * 0.9;

  // 在该段频率两侧各扩展一段搜索
  const fLo = Math.max(0, Math.min(seg.a.f, seg.b.f) - seg.df * 1.5);
  const fHi = Math.min(1, Math.max(seg.a.f, seg.b.f) + seg.df * 1.5);
  const kLo = Math.max(1, Math.floor((fLo * sampleRate / 2 / sampleRate) * resp.nfft));
  const kHi = Math.min(resp.bins - 1, Math.ceil((fHi * sampleRate / 2 / sampleRate) * resp.nfft));

  const crossings = [];
  for (let k = kLo + 1; k <= kHi; k++) {
    const db0 = resp.magDb[k - 1];
    const db1 = resp.magDb[k];
    for (const level of [level10, level90]) {
      if ((db0 - level) * (db1 - level) <= 0 && db1 !== db0) {
        const t = (level - db0) / (db1 - db0);
        const fBin = ((k - 1 + t) * sampleRate) / resp.nfft;
        crossings.push({ level, f: fBin / (sampleRate / 2) });
      }
    }
  }
  if (crossings.length < 2) {
    return { available: false, reason: '实际响应未能在搜索范围内跨过 10%/90% 电平' };
  }
  let f10 = null;
  let f90 = null;
  for (const c of crossings) {
    if (Math.abs(c.level - level10) < 1e-9 && f10 === null) f10 = c.f;
    if (Math.abs(c.level - level90) < 1e-9 && f90 === null) f90 = c.f;
  }
  if (f10 === null || f90 === null) {
    // 退化：取最小/最大交点
    const fs = crossings.map((c) => c.f).sort((a, b) => a - b);
    f10 = fs[0];
    f90 = fs[fs.length - 1];
  }
  const widthNorm = Math.abs(f90 - f10); // 相对 Nyquist
  return {
    available: true,
    widthNormNyquist: widthNorm,
    widthHz: (widthNorm * sampleRate) / 2,
    f10,
    f90,
    level10Db: level10,
    level90Db: level90,
    direction: dir,
  };
}

// 通带纹波 / 阻带衰减：依据目标点的平台粗略分区。
// 阻带最坏电平只在“进入阻带并留出过渡余量”之后统计，避免把过渡沿算成阻带恶化。
export function bandRippleMetrics(resp, targetPoints, sampleRate) {
  let passMax = -Infinity;
  let passMin = Infinity;
  let stopMax = -Infinity;

  let stopStartF = 1;
  for (const p of targetPoints) {
    if (p.db <= -30) {
      stopStartF = Math.min(stopStartF, p.f);
    }
  }
  const stopMeasureF = Math.min(1, stopStartF + 0.03);

  for (let k = 0; k < resp.bins; k++) {
    const f = resp.freqs[k] / (sampleRate / 2);
    const tdb = sampleTargetDb(targetPoints, Math.min(1, Math.max(0, f)));
    if (tdb >= -3) {
      passMax = Math.max(passMax, resp.magDb[k] - tdb);
      passMin = Math.min(passMin, resp.magDb[k] - tdb);
    } else if (tdb <= -30 && f >= stopMeasureF) {
      stopMax = Math.max(stopMax, resp.magDb[k]);
    }
  }
  return {
    passRippleDb: Number.isFinite(passMax) && Number.isFinite(passMin) ? passMax - passMin : null,
    stopFloorDb: Number.isFinite(stopMax) ? stopMax : null,
  };
}

// 截止型过渡宽度：在归一化截止频率 fcNyq（相对 Nyquist）附近，
// 实测响应下降沿穿过 passDb / stopDb 的频率间距。用于长度↔过渡带取舍。
export function measureCutoffTransition(resp, sampleRate, fcNyq = 0.4, passDb = -1, stopDb = -40) {
  const df = sampleRate / resp.nfft;
  const kCenter = Math.round(fcNyq * (resp.bins - 1));
  const kLo = Math.max(1, Math.floor(kCenter * 0.5));
  const kHi = Math.min(resp.bins - 2, Math.ceil(kCenter + (resp.bins - 1) * 0.2));

  // 收集某电平的全部“向高频下行”穿越（线性插值）
  const downwardCrossings = (level) => {
    const out = [];
    for (let k = kLo; k <= kHi; k++) {
      const a = resp.magDb[k];
      const b = resp.magDb[k + 1];
      if (a >= level && b < level && b !== a) {
        const t = (level - a) / (b - a);
        out.push((k + t) * df);
      }
    }
    return out;
  };

  const stopCrossings = downwardCrossings(stopDb);
  const fStop = stopCrossings.length ? stopCrossings[0] : null;
  const passCrossings = downwardCrossings(passDb).filter((f) => f < (fStop ?? Infinity));
  const fPass = passCrossings.length ? passCrossings[passCrossings.length - 1] : null;

  const cross3 = downwardCrossings(-3).filter((f) => f < (fStop ?? Infinity));
  const f3 = cross3.length ? cross3[cross3.length - 1] : null;

  return {
    available: fPass != null && fStop != null && fStop > fPass,
    passEdgeHz: fPass,
    stopEdgeHz: fStop,
    widthHz: fPass != null && fStop != null ? fStop - fPass : null,
    minus3Hz: f3,
    fcNyq,
    passDb,
    stopDb,
  };
}

// 线性相位恒定时延（样本）。偶数 N 时为半整数。
export function nominalGroupDelay(N) {
  return (N - 1) / 2;
}

// 在响应网格上统计群延迟波动（剔除幅度过低的频点）。
export function groupDelayStats(resp, fMin = 20, fMax = 16000) {
  const vals = [];
  for (let k = 0; k < resp.bins; k++) {
    if (!resp.gdValid[k]) continue;
    const f = resp.freqs[k];
    if (f < fMin || f > fMax) continue;
    vals.push(resp.gd[k]);
  }
  if (!vals.length) return { mean: null, maxDeviation: null };
  let mean = 0;
  for (const v of vals) mean += v;
  mean /= vals.length;
  let maxDev = 0;
  for (const v of vals) maxDev = Math.max(maxDev, Math.abs(v - mean));
  return { mean, maxDeviation: maxDev };
}

export { sampleTargetMag };
