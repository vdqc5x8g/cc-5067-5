// 零依赖 radix-2 迭代 FFT。Node 与浏览器同构。

export function nextPow2(n) {
  let p = 1;
  while (p < n) p <<= 1;
  return p;
}

export function isPow2(n) {
  return n >= 1 && (n & (n - 1)) === 0;
}

// 原地 FFT。re/im 为普通 number[] 或 Float64Array，长度必须是 2 的幂。
export function fftInPlace(re, im) {
  const n = re.length;
  if (!isPow2(n)) throw new Error('FFT 长度必须是 2 的幂: ' + n);

  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      let t = re[i]; re[i] = re[j]; re[j] = t;
      t = im[i]; im[i] = im[j]; im[j] = t;
    }
  }

  for (let len = 2; len <= n; len <<= 1) {
    const half = len >> 1;
    const ang = (-2 * Math.PI) / len;
    const wr = Math.cos(ang);
    const wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cwr = 1;
      let cwi = 0;
      for (let k = 0; k < half; k++) {
        const a = i + k;
        const b = a + half;
        const vr = re[b] * cwr - im[b] * cwi;
        const vi = re[b] * cwi + im[b] * cwr;
        re[b] = re[a] - vr;
        im[b] = im[a] - vi;
        re[a] += vr;
        im[a] += vi;
        const nwr = cwr * wr - cwi * wi;
        cwi = cwr * wi + cwi * wr;
        cwr = nwr;
      }
    }
  }
  return { re, im };
}

export function ifftInPlace(re, im) {
  const n = re.length;
  for (let i = 0; i < n; i++) im[i] = -im[i];
  fftInPlace(re, im);
  for (let i = 0; i < n; i++) {
    re[i] = re[i] / n;
    im[i] = -im[i] / n;
  }
  return { re, im };
}

// 实序列 FFT，返回 {re, im}（新数组）。
export function realFft(x, nfft) {
  const n = nfft || nextPow2(x.length);
  const re = new Float64Array(n);
  const im = new Float64Array(n);
  for (let i = 0; i < x.length && i < n; i++) re[i] = x[i];
  fftInPlace(re, im);
  return { re, im, n };
}
