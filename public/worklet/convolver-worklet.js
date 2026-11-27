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
    this.tailRemaining = Math.max(0, this.N - 1);
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

  drainBlock() {
    const out = this.processBlock(new Float32Array(this.B));
    this.tailRemaining = Math.max(0, this.tailRemaining - this.B);
    return out;
  }
}

class LinearPhaseConvolverProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.blockSize = 128;
    this.kernel = null;
    this.pendingKernel = null;
    this.channelStates = [];
    this.kernelSwitches = 0;
    this.flushRequested = false;
    this.port.onmessage = (e) => {
      const msg = e.data || {};
      const kernelMessage = ['kernel', 'switch-kernel', 'replace-kernel', 'kernel-switch', 'hot-swap'].includes(msg.type);
      if (kernelMessage) {
        const kernelData = msg.kernel || msg.newKernel || msg.buffer;
        const k = kernelData instanceof Float32Array ? kernelData : Float32Array.from(kernelData || []);
        if (k.length >= 1) {
          // 不在消息处理里清空旧状态；在下一个 128 量子边界把旧 OLA 状态转入 drains。
          this.pendingKernel = k;
          this.flushRequested = false;
          this.port.postMessage({ type: 'kernel-received', length: k.length, sampleRate });
        }
      } else if (msg.type === 'reset') {
        this.kernel = null;
        this.pendingKernel = null;
        this.channelStates = [];
        this.kernelSwitches = 0;
        this.flushRequested = false;
      } else if (['flush', 'end', 'drain', 'finish'].includes(msg.type)) {
        this.flushRequested = true;
      }
    };
  }

  _applyPendingKernel(outputCount) {
    if (!this.pendingKernel) return;
    this.kernel = this.pendingKernel;
    this.pendingKernel = null;
    while (this.channelStates.length < outputCount) this.channelStates.push({ active: null, drains: [] });
    for (const state of this.channelStates) {
      if (state.active) state.drains.push(state.active);
      state.active = new OLAChannel(this.kernel, this.blockSize);
    }
    this.kernelSwitches++;
    this.port.postMessage({
      type: 'kernel-applied',
      length: this.kernel.length,
      sampleRate,
      switches: this.kernelSwitches,
      blockSize: this.blockSize,
      wholeBlockLatencySamples: this.blockSize,
      hotSwapTail: 'deferred-drain',
    });
  }

  process(inputs, outputs) {
    const output = outputs[0];
    if (!output || !output.length) return true;
    const input = inputs[0] || [];
    this._applyPendingKernel(output.length);

    while (this.channelStates.length < output.length) {
      this.channelStates.push({ active: null, drains: [] });
    }

    for (let c = 0; c < output.length; c++) {
      const out = output[c];
      const state = this.channelStates[c];
      const src = input[Math.min(c, Math.max(0, input.length - 1))];
      const hasSource = src && src.length > 0;

      if (!this.kernel) {
        if (hasSource) out.set(src.length === out.length ? src : src.subarray(0, out.length));
        continue;
      }

      if (hasSource) {
        if (!state.active) state.active = new OLAChannel(this.kernel, this.blockSize);
        out.set(state.active.processBlock(src.length === this.blockSize ? src : src.subarray(0, this.blockSize)));
      } else if (this.flushRequested && state.active) {
        out.set(state.active.drainBlock());
        if (state.active.tailRemaining === 0) state.active = null;
      }

      for (let d = 0; d < state.drains.length; d++) {
        const tail = state.drains[d].drainBlock();
        for (let i = 0; i < out.length; i++) out[i] += tail[i];
      }
      state.drains = state.drains.filter((ch) => ch.tailRemaining > 0);
    }

    if (this.flushRequested && this.channelStates.every((s) => !s.active && !s.drains.length)) {
      this.flushRequested = false;
      this.port.postMessage({
        type: 'tail-flushed',
        event: 'drained',
        drained: true,
        flushComplete: true,
        sampleRate,
      });
    }
    return true;
  }
}

registerProcessor('linear-phase-convolver', LinearPhaseConvolverProcessor);
