// 目标幅度响应：归一化频率 f ∈ [0,1]（1 = Nyquist），dB。
// 这是用户可拖动 EQ 曲线的唯一数据模型，设计器与界面都以它为准。

export const TARGET_LIMITS = {
  minDb: -100,
  maxDb: 24,
  maxPoints: 24,
};

export function clamp(v, lo, hi) {
  return Math.min(hi, Math.max(lo, v));
}

export function defaultTargetPoints() {
  // 低频架式提升 + 高频空气衰减的示例曲线
  return [
    { f: 0.0, db: 4.5 },
    { f: 0.08, db: 4.5 },
    { f: 0.22, db: 1.2 },
    { f: 0.45, db: 0.0 },
    { f: 0.7, db: -1.5 },
    { f: 1.0, db: -1.5 },
  ];
}

export const PRESET_TARGETS = {
  flat: {
    label: '平直',
    points: [
      { f: 0, db: 0 },
      { f: 1, db: 0 },
    ],
  },
  bass: {
    label: '低频架式',
    points: [
      { f: 0, db: 6 },
      { f: 0.1, db: 6 },
      { f: 0.25, db: 0 },
      { f: 1, db: 0 },
    ],
  },
  smile: {
    label: '微笑曲线',
    points: [
      { f: 0, db: 5 },
      { f: 0.12, db: 5 },
      { f: 0.3, db: 0 },
      { f: 0.62, db: 0 },
      { f: 0.85, db: 4 },
      { f: 1, db: 4 },
    ],
  },
  cut: {
    label: '中频削减',
    points: [
      { f: 0, db: 0 },
      { f: 0.2, db: 0 },
      { f: 0.36, db: -6 },
      { f: 0.5, db: -6 },
      { f: 0.66, db: 0 },
      { f: 1, db: 0 },
    ],
  },
  lowpass: {
    label: '陡峭低通目标',
    points: [
      { f: 0, db: 0 },
      { f: 0.3, db: 0 },
      { f: 0.42, db: -3 },
      { f: 0.5, db: -45 },
      { f: 1, db: -60 },
    ],
  },
};

export function normalizePoints(points) {
  return points
    .map((p) => ({ f: Number(p.f), db: Number(p.db) }))
    .sort((a, b) => a.f - b.f)
    .map((p) => ({ f: clamp(p.f, 0, 1), db: clamp(p.db, TARGET_LIMITS.minDb, TARGET_LIMITS.maxDb) }));
}

// 校验“受支持的目标响应”。返回 {ok, error?, message?}。
export function validateTarget(target) {
  if (!target || !Array.isArray(target.points)) {
    return { ok: false, error: 'BAD_TARGET', message: '目标响应缺少 points 数组' };
  }
  const pts = target.points;
  if (pts.length < 2) {
    return { ok: false, error: 'TOO_FEW_POINTS', message: '目标曲线至少需要 2 个控制点' };
  }
  if (pts.length > TARGET_LIMITS.maxPoints) {
    return { ok: false, error: 'TOO_MANY_POINTS', message: `控制点不能超过 ${TARGET_LIMITS.maxPoints} 个` };
  }
  let lastF = -Infinity;
  for (let i = 0; i < pts.length; i++) {
    const p = pts[i];
    if (!Number.isFinite(p.f) || !Number.isFinite(p.db)) {
      return { ok: false, error: 'NON_FINITE', message: `第 ${i + 1} 个控制点不是有限数值` };
    }
    if (p.f < 0 || p.f > 1) {
      return { ok: false, error: 'F_OUT_OF_RANGE', message: `第 ${i + 1} 个点频率超出 [0, Nyquist]` };
    }
    if (p.db < TARGET_LIMITS.minDb || p.db > TARGET_LIMITS.maxDb) {
      return { ok: false, error: 'DB_OUT_OF_RANGE', message: `第 ${i + 1} 个点增益超出允许范围` };
    }
    if (p.f < lastF) {
      return { ok: false, error: 'NOT_SORTED', message: '控制点频率必须单调不减' };
    }
    if (p.f === lastF && i > 0 && pts[i - 1] !== p) {
      // 允许同频双点（阶跃），但数据模型里按线性插值处理时去重
    }
    lastF = p.f;
  }
  return { ok: true };
}

// 在 dB 域分段线性插值。f 超出端点时用端点值（架式延伸）。
export function sampleTargetDb(points, f) {
  const pts = points;
  if (f <= pts[0].f) return pts[0].db;
  if (f >= pts[pts.length - 1].f) return pts[pts.length - 1].db;
  for (let i = 0; i < pts.length - 1; i++) {
    const a = pts[i];
    const b = pts[i + 1];
    if (f >= a.f && f <= b.f) {
      if (b.f === a.f) return b.db;
      const t = (f - a.f) / (b.f - a.f);
      return a.db + t * (b.db - a.db);
    }
  }
  return pts[pts.length - 1].db;
}

export function sampleTargetMag(points, f) {
  return Math.pow(10, sampleTargetDb(points, f) / 20);
}

// 找到曲线中斜率最陡的相邻段，用于过渡带测量。
export function steepestSegment(points) {
  let best = null;
  for (let i = 0; i < points.length - 1; i++) {
    const a = points[i];
    const b = points[i + 1];
    const df = b.f - a.f;
    if (df <= 1e-9) continue;
    const slope = Math.abs(b.db - a.db) / df;
    if (!best || slope > best.slope) best = { i, a, b, df, slope };
  }
  return best;
}
