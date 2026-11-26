// 实时卷积 AudioWorklet：内置 radix-2 FFT 与 overlap-add，逐通道独立状态。
// 与 public/dsp/convolution.js 使用同一算法（离线正确性由 test/ 保证）。
/* global AudioWorkletProcessor, registerProcessor, sampleRate */

function nextPow2(n) {
  let p = 1;
  while (p < n) p <<= 1;
  return p;
}

function fftInPlace(re, im) {
  const n = re.length;
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
      let cwr = 1, cwi = 0;
      for (let k = 0; k < half; k++) {
        const a = i + k, b = a + half;
        const vr = re[b] * cwr - im[b] * cwi;
        const vi = re[b] * cwi + im[b] * cwr;
        re[b] = re[a] - vr; im[b] = im[a] - vi;
        re[a] += vr; im[a] += vi;
        const nwr = cwr * wr - cwi * wi;
        cwi = cwr * wi + cwi * wr;
        cwr = nwr;
      }
    }
  }
}

class OLAChannel {
  constructor(kernel, blockSize) {
    this.h = kernel;
    this.N = kernel.length;
    this.B = blockSize;
    this.L = nextPow2(blockSize + this.N - 1);
    this.ov = new Float64Array(Math.max(0, this.N - 1));
    this.Hr = new Float64Array(this.L);
    this.Hi = new Float64Array(this.L);
    this.Hr.set(kernel);
    fftInPlace(this.Hr, this.Hi);
  }

  processBlock(x) {
    const b = x.length;
    const L = this.L;
    const re = new Float64Array(L);
    const im = new Float64Array(L);
    for (let i = 0; i < b; i++) re[i] = x[i];
    fftInPlace(re, im);
    for (let k = 0; k < L; k++) {
      const hr = this.Hr[k], hi = this.Hi[k];
      const nr = re[k] * hr - im[k] * hi;
      im[k] = re[k] * hi + im[k] * hr;
      re[k] = nr;
    }
    for (let k = 0; k < L; k++) im[k] = -im[k];
    fftInPlace(re, im);
    const out = new Float32Array(b);
    for (let i = 0; i < b; i++) {
      let v = re[i] / L;
      if (i < this.ov.length) v += this.ov[i];
      out[i] = v;
    }
    const next = new Float64Array(this.N - 1);
    for (let i = 0; i < next.length; i++) {
      let v = re[b + i] / L;
      if (b + i < this.ov.length) v += this.ov[b + i];
      next[i] = v;
    }
    this.ov = next;
    return out;
  }
}

class LinearPhaseConvolverProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.blockSize = 128;
    this.kernel = null;
    this.channels = [];
    this.port.onmessage = (e) => {
      const msg = e.data || {};
      if (msg.type === 'kernel') {
        const k = msg.kernel instanceof Float32Array ? msg.kernel : Float32Array.from(msg.kernel || []);
        if (k.length >= 1) {
          this.kernel = k;
          this.channels = [];
          this.port.postMessage({ type: 'kernel-applied', length: k.length, sampleRate });
        }
      } else if (msg.type === 'reset') {
        this.channels = [];
      }
    };
  }

  process(inputs, outputs) {
    const input = inputs[0];
    const output = outputs[0];
    if (!input || !input.length || !output || !output.length) return true;
    if (!this.kernel) {
      for (let c = 0; c < output.length; c++) output[c].set(input[Math.min(c, input.length - 1)] || new Float32Array(this.blockSize));
      return true;
    }
    while (this.channels.length < output.length) {
      this.channels.push(new OLAChannel(this.kernel, this.blockSize));
    }
    for (let c = 0; c < output.length; c++) {
      const src = input[Math.min(c, input.length - 1)];
      if (!src) continue;
      if (!this.channels[c] || this.channels[c].N !== this.kernel.length) {
        this.channels[c] = new OLAChannel(this.kernel, this.blockSize);
      }
      const y = this.channels[c].processBlock(src);
      output[c].set(y);
    }
    return true;
  }
}

registerProcessor('linear-phase-convolver', LinearPhaseConvolverProcessor);
