// 线性相位 FIR 设计器。
// 方法：对目标实振幅做余弦/正弦级数展开（等价于理想无限长核），
// 再乘对称窗截断。抽头按所选 FIR 类型的对称关系直接构造，
// 因此“所选对称条件”由构造保证，而不是事后声称。
//
// 这不是双二阶（Biquad）串联：输出是有限长对称 FIR，
// 群延迟恒为 (N-1)/2（I/III 整数，II/IV 半整数）。

import { makeWindow, kaiserBetaForStopband } from './windows.js';
import {
  validateTarget,
  sampleTargetMag,
  clamp,
} from './target.js';
import {
  frequencyResponse,
  approximationError,
  measureTransitionBand,
  measureCutoffTransition,
  bandRippleMetrics,
  symmetryResidual,
  nominalGroupDelay,
  groupDelayStats,
} from './response.js';

export const FIR_TYPES = {
  I: { roman: 'I', parity: 'odd', symmetry: 'symmetric', desc: '奇数长 · 偶对称（通用 EQ，支持 Nyquist）' },
  II: { roman: 'II', parity: 'even', symmetry: 'symmetric', desc: '偶数长 · 偶对称（Nyquist 必为零点）' },
  III: { roman: 'III', parity: 'odd', symmetry: 'antisymmetric', desc: '奇数长 · 奇对称（微分/Hilbert，0 与 Nyquist 均为零）' },
  IV: { roman: 'IV', parity: 'even', symmetry: 'antisymmetric', desc: '偶数长 · 奇对称（微分器，直流为零）' },
};

export function normalizeLength(N, firType) {
  let n = Math.max(3, Math.floor(Number(N) || 127));
  const needOdd = firType === 'I' || firType === 'III';
  const isOdd = n % 2 === 1;
  if (needOdd && !isOdd) n += 1;
  if (!needOdd && isOdd) n += 1;
  return Math.min(n, 4097);
}

// 数值积分提取级数系数。kind: 'cos' | 'sin'，half: 是否半整数阶。
function extractCoeffs(targetAmplitude, count, kind, half, gridJ = 8192) {
  const c = new Float64Array(count);
  for (let k = 1; k <= count; k++) {
    const order = half ? k - 0.5 : k;
    let sum = 0;
    for (let j = 0; j <= gridJ; j++) {
      const w = (Math.PI * j) / gridJ;
      const A = targetAmplitude(w);
      const basis = kind === 'cos' ? Math.cos(order * w) : Math.sin(order * w);
      let weight = 1;
      if (j === 0 || j === gridJ) weight = 0.5;
      sum += weight * A * basis;
    }
    c[k - 1] = sum / gridJ; // 1/π × 梯形积分(步长 π/J)
  }
  return c;
}

function eqAmplitude(points) {
  return (omega) => sampleTargetMag(points, omega / Math.PI);
}

function differentiatorAmplitude(omega) {
  // 理想微分器在 0..π 的实振幅（线性上升）
  return omega / Math.PI;
}

// 检查 FIR 类型对目标端点的固有限制。
export function checkTypeCompatibility(firType, targetPoints) {
  const at0 = sampleTargetMag(targetPoints, 0);
  const at1 = sampleTargetMag(targetPoints, 1);
  const loud = (m) => 20 * Math.log10(Math.max(m, 1e-8)) > -36;
  if (firType === 'II' && loud(at1)) {
    return {
      ok: false,
      code: 'TYPE2_NYQUIST_ZERO',
      message: 'II 型（偶数长偶对称）FIR 在 Nyquist 频率必有零点，无法实现当前 Nyquist 处的目标增益。请改用 I 型（奇数长）。',
      suggestion: { firType: 'I' },
    };
  }
  if (firType === 'III' && (loud(at0) || loud(at1))) {
    return {
      ok: false,
      code: 'TYPE3_ENDPOINT_ZEROS',
      message: 'III 型 FIR 在直流与 Nyquist 都有零点，不能实现该目标。EQ 请使用 I 型。',
      suggestion: { firType: 'I' },
    };
  }
  if (firType === 'IV' && loud(at0)) {
    return {
      ok: false,
      code: 'TYPE4_DC_ZERO',
      message: 'IV 型 FIR 在直流必有零点，不能实现带直流增益的 EQ 目标。请改用 I 型或 II 型。',
      suggestion: { firType: 'I' },
    };
  }
  return { ok: true };
}

/**
 * 设计线性相位 FIR。
 * options:
 *   points: [{f,db}] 目标曲线（与 target 二选一）
 *   target: {points}
 *   N, firType ('I'..'IV'), window, kaiserBeta, stopbandDb
 *   sampleRate, nfft
 *   kind: 'eq' | 'differentiator'
 *   specs: {maxTransitionHz?, maxErrorDb?} 可选规格判定
 */
export function designFir(options = {}) {
  const sampleRate = options.sampleRate || 48000;
  const firType = options.firType || 'I';
  const typeInfo = FIR_TYPES[firType];
  if (!typeInfo) {
    return { ok: false, error: 'BAD_TYPE', message: '未知 FIR 类型: ' + firType };
  }
  const N = normalizeLength(options.N || 127, firType);
  const windowId = options.window || 'hann';
  const kaiserBeta =
    options.kaiserBeta != null
      ? options.kaiserBeta
      : kaiserBetaForStopband(options.stopbandDb ?? 60);

  let points = options.points || (options.target && options.target.points);
  const kind = options.kind || 'eq';

  if (kind === 'eq') {
    const v = validateTarget({ points });
    if (!v.ok) return { ok: false, error: v.error, message: v.message };
    points = points.map((p) => ({ f: clamp(Number(p.f), 0, 1), db: Number(p.db) }));
    const compat = checkTypeCompatibility(firType, points);
    if (!compat.ok) {
      return { ok: false, error: compat.code, message: compat.message, suggestion: compat.suggestion, N, firType };
    }
  }

  const gridJ = options.gridJ || Math.max(2048, Math.min(16384, N * 32));
  const h = new Float64Array(N);

  try {
    if (firType === 'I') {
      const M = (N - 1) / 2;
      const A = kind === 'differentiator' ? differentiatorAmplitude : eqAmplitude(points);
      // c0 单独用端点含半权的梯形求和
      let s0 = 0;
      for (let j = 0; j <= gridJ; j++) {
        const w = (Math.PI * j) / gridJ;
        const weight = j === 0 || j === gridJ ? 0.5 : 1;
        s0 += weight * A(w);
      }
      const c0 = s0 / gridJ;
      const ck = extractCoeffs(A, M, 'cos', false, gridJ);
      h[M] = c0;
      for (let k = 1; k <= M; k++) {
        h[M - k] = ck[k - 1];
        h[M + k] = ck[k - 1];
      }
    } else if (firType === 'II') {
      const L = N / 2;
      const A = eqAmplitude(points);
      const bk = extractCoeffs(A, L, 'cos', true, gridJ);
      for (let k = 1; k <= L; k++) {
        h[L - k] = bk[k - 1];
        h[L - 1 + k] = bk[k - 1];
      }
    } else if (firType === 'III') {
      const M = (N - 1) / 2;
      const A = differentiatorAmplitude;
      const ck = extractCoeffs(A, M, 'sin', false, gridJ);
      h[M] = 0;
      for (let k = 1; k <= M; k++) {
        h[M - k] = ck[k - 1];
        h[M + k] = -ck[k - 1];
      }
    } else {
      const L = N / 2;
      const A = differentiatorAmplitude;
      const ck = extractCoeffs(A, L, 'sin', true, gridJ);
      for (let k = 1; k <= L; k++) {
        h[L - k] = ck[k - 1];
        h[L - 1 + k] = -ck[k - 1];
      }
    }
  } catch (err) {
    return { ok: false, error: 'DESIGN_EXCEPTION', message: String(err && err.message || err) };
  }

  const win = makeWindow(windowId, N, kaiserBeta);
  for (let n = 0; n < N; n++) h[n] *= win[n];

  // 数值健全性
  let finite = true;
  let peak = 0;
  for (const v of h) {
    if (!Number.isFinite(v)) finite = false;
    peak = Math.max(peak, Math.abs(v));
  }
  if (!finite || peak === 0) {
    return { ok: false, error: 'KERNEL_INVALID', message: '设计出的核包含非有限值或全零，目标响应可能不受支持' };
  }

  const resp = frequencyResponse(h, sampleRate, options.nfft);
  const sym = symmetryResidual(h);
  const gdNominal = nominalGroupDelay(N);
  const gdStats = groupDelayStats(resp);
  const approx = kind === 'eq' ? approximationError(resp, points, sampleRate) : null;
  const trans = kind === 'eq' ? measureTransitionBand(resp, points, sampleRate) : null;
  const cutoffTransition =
    kind === 'eq' && options.cutoffNyq != null
      ? measureCutoffTransition(
          resp,
          sampleRate,
          options.cutoffNyq,
          options.passDb ?? -1,
          options.stopDb ?? -40
        )
      : null;
  const ripple = kind === 'eq' ? bandRippleMetrics(resp, points, sampleRate) : null;

  let specs = null;
  if (options.specs && kind === 'eq') {
    const transitionForSpec = options.cutoffNyq != null ? cutoffTransition : trans;
    const transitionHz =
      transitionForSpec && transitionForSpec.available
        ? transitionForSpec.widthHz
        : null;
    const checks = [];
    if (options.specs.maxTransitionHz != null) {
      checks.push({
        id: 'transition',
        label: `过渡带 ≤ ${options.specs.maxTransitionHz} Hz`,
        pass: transitionHz != null && transitionHz <= options.specs.maxTransitionHz,
        actual: transitionHz,
      });
    }
    if (options.specs.maxErrorDb != null && approx.maxErrorDb != null) {
      checks.push({
        id: 'error',
        label: `最大误差 ≤ ${options.specs.maxErrorDb} dB`,
        pass: approx.maxErrorDb <= options.specs.maxErrorDb,
        actual: approx.maxErrorDb,
      });
    }
    specs = { checks, pass: checks.length ? checks.every((c) => c.pass) : false };
  }

  return {
    ok: true,
    kernel: Array.from(h),
    N,
    firType,
    symmetry: typeInfo.symmetry,
    typeDesc: typeInfo.desc,
    window: windowId,
    kaiserBeta: windowId === 'kaiser' ? kaiserBeta : null,
    sampleRate,
    groupDelaySamples: gdNominal,
    latencyMs: (gdNominal / sampleRate) * 1000,
    tailSamples: N - 1,
    tailMs: ((N - 1) / sampleRate) * 1000,
    symmetryResidual: sym,
    groupDelay: { nominal: gdNominal, measuredMean: gdStats.mean, maxDeviationSamples: gdStats.maxDeviation },
    approximation: approx
      ? { maxErrorDb: approx.maxErrorDb, rmsErrorDb: approx.rmsErrorDb }
      : null,
    transition: trans,
    cutoffTransition,
    ripple,
    specs,
    response: {
      nfft: resp.nfft,
      freqs: Array.from(resp.freqs),
      magDb: Array.from(resp.magDb),
      phaseUnwrapped: Array.from(resp.phaseUnwrapped),
      gd: Array.from(resp.gd, (v, k) => (resp.gdValid[k] ? v : null)),
    },
  };
}

// 长度/过渡带/误差取舍扫描。
// 每行同时给出：
//   - 砖墙截止规格（specCutoffNyq）下实测 -1/-40dB 过渡宽度，体现 Δf ∝ fs/N；
//   - 对用户目标（points）重设计的实测逼近误差/纹波。
export function scanLengthTradeoff(points, opts = {}) {
  const lengths = opts.lengths || [33, 65, 127, 255, 511, 1023];
  const fc = opts.specCutoffNyq ?? 0.35;
  const sampleRate = opts.sampleRate || 48000;
  const common = {
    firType: 'I',
    window: opts.window || 'kaiser',
    kaiserBeta: opts.kaiserBeta,
    stopbandDb: opts.stopbandDb ?? 70,
    sampleRate,
  };
  const brickwallPoints = [
    { f: 0, db: 0 },
    { f: fc, db: 0 },
    { f: Math.min(1, fc + 1e-4), db: -90 },
    { f: 1, db: -90 },
  ];
  const rows = [];
  for (const N0 of lengths) {
    const N = normalizeLength(N0, 'I');
    const row = { N, ok: true, groupDelaySamples: (N - 1) / 2, latencyMs: (((N - 1) / 2) / sampleRate) * 1000 };

    const bw = designFir({ ...common, N, points: brickwallPoints, cutoffNyq: fc });
    if (bw.ok && bw.cutoffTransition && bw.cutoffTransition.available) {
      row.transitionHz = bw.cutoffTransition.widthHz;
      row.transition3dbHz = bw.cutoffTransition.minus3Hz;
      row.stopFloorDb = bw.ripple.stopFloorDb;
    } else {
      row.transitionHz = null;
    }

    if (points && points.length >= 2) {
      const usr = designFir({ ...common, N, points, specs: opts.specs });
      if (usr.ok) {
        row.maxErrorDb = usr.approximation.maxErrorDb;
        row.rmsErrorDb = usr.approximation.rmsErrorDb;
        row.passRippleDb = usr.ripple.passRippleDb;
        row.specPass = usr.specs ? usr.specs.pass : null;
      } else {
        row.userError = usr.message;
      }
    }
    rows.push(row);
  }
  return rows;
}
