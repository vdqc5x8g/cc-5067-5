// 真实卷积：直接卷积参照 + overlap-add 分块流式卷积。
// StreamingConvolver.processAll 的结果必须与 convolveDirect 相符（见 test/）。

import { nextPow2 } from './fft.js';
import { fftInPlace } from './fft.js';

export function convolveDirect(x, h) {
  const n = x.length;
  const m = h.length;
  if (n === 0 || m === 0) return new Float64Array(0);
  const y = new Float64Array(n + m - 1);
  for (let i = 0; i < n; i++) {
    const xi = x[i];
    if (xi === 0) continue;
    for (let j = 0; j < m; j++) {
      y[i + j] += xi * h[j];
    }
  }
  return y;
}

// 多通道辅助
export function convolveChannels(channels, h) {
  return channels.map((ch) => convolveDirect(ch, h));
}

export class OverlapAddConvolver {
  // blockSize: 每次 push 产出的样本块大小（与 AudioWorklet 128 对齐即可）
  constructor(kernel, blockSize = 128) {
    this.h = kernel instanceof Float64Array ? kernel : Float64Array.from(kernel);
    this.N = this.h.length;
    this.blockSize = blockSize;
    this.fftSize = nextPow2(blockSize + this.N - 1);
    this.H = this._kernelSpectrum();
    this.overlap = new Float64Array(Math.max(0, this.N - 1));
    this._inAcc = new Float64Array(0);
    this.totalInput = 0;
    this.totalOutput = 0;
  }

  _kernelSpectrum() {
    const L = this.fftSize;
    const re = new Float64Array(L);
    const im = new Float64Array(L);
    re.set(this.h);
    fftInPlace(re, im);
    return { re, im };
  }

  _convolveBlockAdd(xb) {
    // xb: 长度 b（<= blockSize）。返回长度 b 输出，并更新 overlap。
    const b = xb.length;
    const L = this.fftSize;
    const re = new Float64Array(L);
    const im = new Float64Array(L);
    re.set(xb);
    fftInPlace(re, im);
    for (let k = 0; k < L; k++) {
      const hr = this.H.re[k];
      const hi = this.H.im[k];
      const nr = re[k] * hr - im[k] * hi;
      const ni = re[k] * hi + im[k] * hr;
      re[k] = nr;
      im[k] = ni;
    }
    // 逆 FFT（共轭法）
    for (let k = 0; k < L; k++) im[k] = -im[k];
    fftInPlace(re, im);
    const y = new Float64Array(L);
    for (let k = 0; k < L; k++) y[k] = re[k] / L;

    const out = new Float64Array(b);
    const oldOv = this.overlap;
    for (let i = 0; i < b; i++) out[i] = y[i] + (i < oldOv.length ? oldOv[i] : 0);

    const newLen = this.N - 1;
    const next = new Float64Array(newLen);
    for (let i = 0; i < newLen; i++) {
      let v = y[b + i] || 0;
      if (b + i < oldOv.length) v += oldOv[b + i];
      next[i] = v;
    }
    this.overlap = next;
    return out;
  }

  // 送入任意长度输入，返回当前可输出的完整 blockSize 块（拼接好的连续样本）。
  push(chunk) {
    const merged = new Float64Array(this._inAcc.length + chunk.length);
    merged.set(this._inAcc, 0);
    merged.set(chunk, this._inAcc.length);
    this._inAcc = merged;

    const blocks = [];
    let produced = 0;
    while (this._inAcc.length >= this.blockSize) {
      const xb = this._inAcc.subarray(0, this.blockSize);
      const out = this._convolveBlockAdd(xb);
      blocks.push(out);
      produced += out.length;
      this._inAcc = this._inAcc.slice(this.blockSize);
    }
    this.totalInput += chunk.length;
    if (produced) this.totalOutput += produced;
    if (!blocks.length) return new Float64Array(0);
    const all = new Float64Array(produced);
    let off = 0;
    for (const b of blocks) { all.set(b, off); off += b.length; }
    return all;
  }

  // 结束流：处理残余输入块，再放出 N-1 个重叠尾部。
  flush() {
    const pieces = [];
    if (this._inAcc.length > 0) {
      const rest = this._inAcc;
      const out = this._convolveBlockAdd(rest);
      pieces.push(out);
      this.totalOutput += out.length;
      this._inAcc = new Float64Array(0);
    }
    if (this.overlap.length) {
      pieces.push(this.overlap.slice());
      this.totalOutput += this.overlap.length;
    }
    const len = pieces.reduce((a, p) => a + p.length, 0);
    const all = new Float64Array(len);
    let off = 0;
    for (const p of pieces) { all.set(p, off); off += p.length; }
    return all;
  }

  // 一次性流式处理任意输入，结果长度恰为 x.length + N - 1。
  processAll(x) {
    const head = this.push(x);
    const tail = this.flush();
    const y = new Float64Array(head.length + tail.length);
    y.set(head, 0);
    y.set(tail, head.length);
    return y;
  }
}

// 不依赖类的便捷封装（每次新建状态）
export function streamConvolve(x, h, blockSize = 128) {
  const c = new OverlapAddConvolver(h, blockSize);
  return c.processAll(x);
}

// 多通道实时/离线处理（各通道独立流式状态，核相同）
export function processChannelsStreaming(channels, h, blockSize = 128) {
  return channels.map((ch) => streamConvolve(ch, h, blockSize));
}
