// 对称窗函数。所有窗均关于中点对称，加窗不会破坏 FIR 的对称条件。

function besselI0(x) {
  let sum = 1;
  let term = 1;
  const x2 = (x / 2) * (x / 2);
  for (let k = 1; k < 64; k++) {
    term *= x2 / (k * k);
    sum += term;
    if (term < 1e-14 * sum) break;
  }
  return sum;
}

// Kaiser beta 与阻带衰减 A(dB) 的经验关系。
export function kaiserBetaForStopband(attenDb) {
  const A = Math.max(0, attenDb);
  if (A < 21) return 0;
  if (A <= 50) return 0.5842 * Math.pow(A - 21, 0.4) + 0.07886 * (A - 21);
  return 0.1102 * (A - 8.7);
}

export const WINDOW_CHOICES = [
  { id: 'hann', label: 'Hann（汉宁）', broadening: 3.1 },
  { id: 'hamming', label: 'Hamming（汉明）', broadening: 3.3 },
  { id: 'blackman', label: 'Blackman（布莱克曼）', broadening: 5.5 },
  { id: 'kaiser', label: 'Kaiser（凯泽）', broadening: null },
  { id: 'rectangular', label: '矩形（截断）', broadening: 0.9 },
];

// 返回长度 N 的对称窗。kaiserBeta 仅 kaiser 使用。
export function makeWindow(id, N, kaiserBeta = 8.6) {
  const w = new Float64Array(N);
  if (N === 1) {
    w[0] = 1;
    return w;
  }
  const mid = (N - 1) / 2;
  const den = N - 1;
  const i0beta = id === 'kaiser' ? besselI0(kaiserBeta) : 1;
  for (let n = 0; n < N; n++) {
    const x = (2 * n) / den - 1; // -1..1
    switch (id) {
      case 'rectangular':
        w[n] = 1;
        break;
      case 'hann':
        w[n] = 0.5 - 0.5 * Math.cos((2 * Math.PI * n) / den);
        break;
      case 'hamming':
        w[n] = 0.54 - 0.46 * Math.cos((2 * Math.PI * n) / den);
        break;
      case 'blackman':
        w[n] =
          0.42 -
          0.5 * Math.cos((2 * Math.PI * n) / den) +
          0.08 * Math.cos((4 * Math.PI * n) / den);
        break;
      case 'kaiser': {
        const arg = Math.sqrt(Math.max(0, 1 - x * x));
        w[n] = besselI0(kaiserBeta * arg) / i0beta;
        break;
      }
      default:
        throw new Error('未知窗类型: ' + id);
    }
    void mid;
  }
  return w;
}
