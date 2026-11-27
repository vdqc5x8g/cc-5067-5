// FIR 卷积后端适配层。
// 该模块不把后端自报延迟当成事实：报告值只用于展示，对齐与验收始终回到
// 直接卷积参照、实测群延迟和完整 x+N-1 尾部。

import { OverlapAddConvolver, convolveDirect } from './convolution.js';
import { designFir } from './fir-design.js';
import { findPeak } from './audio.js';

export const FIR_BACKEND_API_VERSION = '2.0.0';
export const CONVOLUTION_BACKEND_VERSION = 'ola-js-2.0.0';
export const CONFIG_SCHEMA_VERSION = 2;

export const BACKEND_BUILD_LOCK = Object.freeze({
  package: 'linear-phase-eq',
  packageVersion: '1.1.0',
  lockfileVersion: 3,
  convolutionBackend: 'overlap-add-js',
  convolutionBackendVersion: CONVOLUTION_BACKEND_VERSION,
  firBackendApiVersion: FIR_BACKEND_API_VERSION,
  runtimeDependencyCount: 0,
});

export const TOLERANCE_PROFILES = Object.freeze({
  float64: Object.freeze({ precision: 'float64', maxAbs: 1e-8, rms: 1e-9, relativeRms: 1e-9, phaseCoherence: 0.999999 }),
  float32: Object.freeze({ precision: 'float32', maxAbs: 2e-5, rms: 5e-6, relativeRms: 2e-5, phaseCoherence: 0.9999 }),
  wasm: Object.freeze({ precision: 'wasm-float32', maxAbs: 5e-5, rms: 1e-5, relativeRms: 5e-5, phaseCoherence: 0.999 }),
});

const ALIAS_CONVENTIONS = {
  deferred: 'deferred-drain',
  drain: 'deferred-drain',
  'deferred-tail': 'deferred-drain',
  'long-partition': 'deferred-drain',
  'partition-drain': 'deferred-drain',
  'long-partition-drain': 'deferred-drain',
  merged: 'merged-final',
  merge: 'merged-final',
  'final-block': 'merged-final',
  'last-block-merged': 'merged-final',
  'merge-final-block': 'merged-final',
  'merged-last-block': 'merged-final',
  'merged-final-block': 'merged-final',
  flush: 'single-flush',
  'single-flush': 'single-flush',
};

export function normalizeTailConvention(value, fallback = 'auto') {
  if (!value) return fallback;
  const key = String(value).trim().toLowerCase().replace(/[\s_]+/g, '-');
  return ALIAS_CONVENTIONS[key] || (key === 'auto' ? 'auto' : fallback);
}

function toArray(chunk) {
  if (chunk == null) return new Float64Array(0);
  if (chunk instanceof Float64Array) return chunk;
  if (ArrayBuffer.isView(chunk)) return Float64Array.from(chunk);
  if (Array.isArray(chunk)) return Float64Array.from(chunk);
  if (chunk && typeof chunk.length === 'number') return Float64Array.from(chunk);
  return new Float64Array(0);
}

function concatArrays(parts) {
  const length = parts.reduce((n, p) => n + p.length, 0);
  const out = new Float64Array(length);
  let offset = 0;
  for (const p of parts) {
    out.set(p, offset);
    offset += p.length;
  }
  return out;
}

function readMaybeFunction(obj, names, ...args) {
  for (const name of names) {
    const value = obj?.[name];
    if (typeof value === 'function') return { name, value: value.apply(obj, args) };
  }
  return null;
}

function readNumberField(obj, names) {
  for (const name of names) {
    const value = obj?.[name];
    if (Number.isFinite(Number(value))) return Number(value);
    if (typeof value === 'function') {
      try {
        const v = value.call(obj);
        if (Number.isFinite(Number(v))) return Number(v);
      } catch { /* 后端报告接口异常时走默认值 */ }
    }
  }
  return null;
}

export function getLatencyReport(backend, fallback = {}) {
  const blockSize = readNumberField(backend, ['blockSize', 'quantum', 'blockSamples']) || fallback.blockSize || 128;
  const kernel = backend?.kernel || fallback.kernel;
  const N = backend?.N || kernel?.length || fallback.kernelLength || 0;
  const nominalGd = N ? (N - 1) / 2 : fallback.groupDelaySamples ?? null;
  const nestedLatency = backend?.latency && typeof backend.latency === 'object' ? backend.latency : null;
  const reported = readNumberField(backend, [
    'latencySamples', 'reportedLatencySamples', 'outputLatencySamples',
    'blockLatencySamples', 'latency', 'delaySamples',
  ]) ?? readNumberField(nestedLatency, [
    'latencySamples', 'samples', 'reportedLatencySamples', 'outputLatencySamples', 'delaySamples',
  ]);
  const wholeBlock = readNumberField(backend, ['wholeBlockLatencySamples', 'blockLatencySamples'])
    ?? readNumberField(nestedLatency, ['wholeBlockLatencySamples', 'wholeBlockSamples', 'blockLatencySamples', 'blockSamples']);
  let detailed = null;
  const reportCall = readMaybeFunction(backend, ['getLatencyReport', 'latencyReport', 'reportLatency', 'getLatency']);
  if (reportCall?.value && typeof reportCall.value === 'object') detailed = reportCall.value;

  return {
    blockSize,
    reportedLatencySamples: reported ?? detailed?.reportedLatencySamples ?? null,
    wholeBlockLatencySamples: wholeBlock ?? detailed?.wholeBlockLatencySamples ?? reported ?? blockSize,
    algorithmicGroupDelaySamples: nominalGd,
    groupDelaySamples: detailed?.groupDelaySamples ?? nominalGd,
    tailSamples: N ? N - 1 : detailed?.tailSamples ?? null,
    precision: backend?.precision || fallback.precision || 'float64',
    raw: detailed,
  };
}

// 尾部按长分区延迟排空的参考后端。push/write 只返回完整块；end 取最后有效
// 部分块，drain 可分多次取出长尾部。flush 保持 OLA 风格的一次性总排空。
export class DeferredTailConvolutionBackend {
  constructor(kernel, options = {}) {
    this.kernel = kernel instanceof Float64Array ? kernel : Float64Array.from(kernel || []);
    this.N = this.kernel.length;
    this.blockSize = options.blockSize || 128;
    this.precision = options.precision || 'float64';
    this.tailConvention = 'deferred-drain';
    this.version = CONVOLUTION_BACKEND_VERSION;
    this.latencySamples = options.latencySamples ?? this.blockSize;
    this.wholeBlockLatencySamples = this.blockSize;
    this.drainChunkSize = options.drainChunkSize || this.blockSize;
    this._ola = new OverlapAddConvolver(this.kernel, this.blockSize);
    this._tail = null;
    this._finalOutput = null;
    this.ended = false;
  }

  latencyReport() {
    const gd = this.N ? (this.N - 1) / 2 : null;
    return {
      blockSize: this.blockSize,
      wholeBlockLatencySamples: this.wholeBlockLatencySamples,
      reportedLatencySamples: this.latencySamples ?? this.wholeBlockLatencySamples,
      algorithmicGroupDelaySamples: gd,
      groupDelaySamples: gd,
      tailSamples: this.N ? this.N - 1 : 0,
      precision: this.precision,
      tailConvention: this.tailConvention,
      version: this.version,
    };
  }

  write(chunk) { return this._ola.push(toArray(chunk)); }
  push(chunk) { return this.write(chunk); }
  process(chunk) { return this.write(chunk); }
  processChunk(chunk) { return this.write(chunk); }

  end(chunk) {
    if (this.ended) return new Float64Array(0);
    this.ended = true;
    const parts = [];
    if (chunk && chunk.length) parts.push(this._ola.push(toArray(chunk)));
    if (this._ola._inAcc.length > 0) {
      const rest = this._ola._inAcc.slice();
      const last = this._ola._convolveBlockAdd(rest);
      this._ola._inAcc = new Float64Array(0);
      this._finalOutput = last;
      parts.push(last);
    } else {
      this._finalOutput = new Float64Array(0);
    }
    this._tail = this._ola.overlap.slice();
    return concatArrays(parts);
  }

  finish(chunk) { return this.end(chunk); }

  drain(maxSamples = this.drainChunkSize) {
    if (!this.ended) this.end();
    if (!this._tail || this._tail.length === 0) return new Float64Array(0);
    if (maxSamples === Infinity || maxSamples === 'all' || maxSamples >= this._tail.length) {
      const all = this._tail;
      this._tail = new Float64Array(0);
      return all;
    }
    const n = Math.max(1, Math.min(maxSamples | 0, this._tail.length));
    const out = this._tail.slice(0, n);
    this._tail = this._tail.slice(n);
    return out;
  }

  drainAll() { return this.drain(Infinity); }

  flush() {
    const last = this.end();
    const tail = this.drain(Infinity);
    return concatArrays([last, tail]);
  }

  processAll(x) {
    return concatArrays([this.write(x), this.flush()]);
  }
}

export class LongPartitionDrainConvolver extends DeferredTailConvolutionBackend {}
export class DeferredPartitionBackend extends DeferredTailConvolutionBackend {}
export class DelayedDrainConvolver extends DeferredTailConvolutionBackend {}

// 最后有效块与尾部合并返回的参考后端。它会保留最后一个完整块，在
// finish/end/flush 中一次性返回“最后有效块 + 卷积尾”，之后不得再 drain。
export class MergedFinalBlockConvolutionBackend {
  constructor(kernel, options = {}) {
    this.kernel = kernel instanceof Float64Array ? kernel : Float64Array.from(kernel || []);
    this.N = this.kernel.length;
    this.blockSize = options.blockSize || 128;
    this.precision = options.precision || 'float64';
    this.tailConvention = 'merged-final';
    this.version = CONVOLUTION_BACKEND_VERSION;
    this.latencySamples = options.latencySamples ?? this.blockSize;
    this.wholeBlockLatencySamples = this.blockSize;
    this._ola = new OverlapAddConvolver(this.kernel, this.blockSize);
    this._pending = new Float64Array(0);
    this.finished = false;
  }

  latencyReport() {
    const gd = this.N ? (this.N - 1) / 2 : null;
    return {
      blockSize: this.blockSize,
      wholeBlockLatencySamples: this.wholeBlockLatencySamples,
      reportedLatencySamples: this.latencySamples ?? this.wholeBlockLatencySamples,
      algorithmicGroupDelaySamples: gd,
      groupDelaySamples: gd,
      tailSamples: this.N ? this.N - 1 : 0,
      precision: this.precision,
      tailConvention: this.tailConvention,
      version: this.version,
    };
  }

  _releaseExceptLastBlock(chunk) {
    this._pending = concatArrays([this._pending, chunk]);
    if (this._pending.length <= this.blockSize) return new Float64Array(0);
    const releaseLength = this._pending.length - this.blockSize;
    const release = this._pending.slice(0, releaseLength);
    this._pending = this._pending.slice(releaseLength);
    return release;
  }

  write(chunk) {
    const produced = this._ola.push(toArray(chunk));
    return this._releaseExceptLastBlock(produced);
  }

  push(chunk) { return this.write(chunk); }
  process(chunk) { return this.write(chunk); }
  processChunk(chunk) { return this.write(chunk); }

  end(chunk) {
    if (this.finished) return new Float64Array(0);
    this.finished = true;
    if (chunk && chunk.length) {
      const produced = this._ola.push(toArray(chunk));
      this._pending = concatArrays([this._pending, produced]);
    }

    let lastValid = new Float64Array(0);
    if (this._ola._inAcc.length > 0) {
      const rest = this._ola._inAcc.slice();
      lastValid = this._ola._convolveBlockAdd(rest);
      this._ola._inAcc = new Float64Array(0);
    }

    const tail = this._ola.overlap.slice();
    const result = concatArrays([this._pending, lastValid, tail]);
    this._pending = new Float64Array(0);
    return result;
  }

  finish(chunk) { return this.end(chunk); }
  flush(chunk) { return this.end(chunk); }

  processAll(x) {
    const head = this.write(x);
    const last = this.finish();
    return concatArrays([head, last]);
  }
}

export class MergedFinalBlockBackend extends MergedFinalBlockConvolutionBackend {}
export class FinalBlockMergedConvolver extends MergedFinalBlockConvolutionBackend {}
export class BlockMergedTailConvolver extends MergedFinalBlockConvolutionBackend {}

function appendOutput(adapter, value) {
  const arr = toArray(value);
  if (arr.length) {
    adapter.parts.push(arr);
    adapter.outputLength += arr.length;
  }
  return arr;
}

export class FirBackendAdapter {
  constructor(backend, options = {}) {
    this.backend = backend && typeof backend === 'object' ? backend : null;
    this.BackendClass = typeof backend === 'function' ? backend : options.BackendClass || null;
    this.options = options;
    this.kernel = options.kernel || this.backend?.kernel || null;
    this.blockSize = options.blockSize || this.backend?.blockSize || 128;
    if (!this.backend && this.BackendClass) this._instantiate();
    this.tailConvention = normalizeTailConvention(
      options.tailConvention || this.backend?.tailConvention || this.backend?.flushConvention || this.backend?.tailMode,
      'auto'
    );
    this.precision = options.precision || this.backend?.precision || 'float64';
    this.tolerance = { ...(TOLERANCE_PROFILES[this.precision] || TOLERANCE_PROFILES.float64), ...(options.tolerance || {}) };
    this.latency = getLatencyReport(this.backend || {}, {
      kernel: this.kernel,
      blockSize: this.blockSize,
      precision: this.precision,
    });
    if (this.tailConvention === 'auto') this.tailConvention = this._inferConvention();
    this.parts = [];
    this.outputLength = 0;
    this.ended = false;
    this.tailCalls = [];
    this.tailBuffer = new Float64Array(0);
    this.tailReadOffset = 0;
  }

  _inferConvention() {
    const b = this.backend;
    if (!b) return 'single-flush';
    const explicit = normalizeTailConvention(b.tailConvention || b.flushConvention || b.tailMode, 'auto');
    if (explicit !== 'auto') return explicit;
    if (typeof b.drain === 'function' || typeof b.drainAll === 'function' || typeof b.readTail === 'function') {
      return 'deferred-drain';
    }
    if (typeof b.finish === 'function' || typeof b.end === 'function') return 'merged-final';
    return 'single-flush';
  }

  _instantiate() {
    if (this.backend) return this.backend;
    if (!this.BackendClass) throw new Error('FirBackendAdapter 需要后端实例、类或 kernel');
    this.backend = new this.BackendClass(this.kernel, { ...this.options, blockSize: this.blockSize });
    return this.backend;
  }

  push(chunk) {
    const b = this._instantiate();
    const result = readMaybeFunction(b, ['push', 'write', 'processChunk', 'processBlock', 'process'], chunk);
    let out = result?.value;
    if (out == null && typeof b.read === 'function') out = b.read();
    return appendOutput(this, out);
  }

  write(chunk) { return this.push(chunk); }

  _callTerminal(names, ...args) {
    const b = this._instantiate();
    const found = readMaybeFunction(b, names, ...args);
    if (found) this.tailCalls.push(found.name);
    return found?.value;
  }

  end(chunk) {
    if (this.ended) return new Float64Array(0);
    this.ended = true;
    const b = this._instantiate();

    if (this.tailConvention === 'merged-final') {
      const out = this._callTerminal(['finish', 'end', 'flush'], chunk);
      return appendOutput(this, out);
    }

    if (this.tailConvention === 'deferred-drain') {
      const last = this._callTerminal(['end', 'finish', 'finishInput'], chunk);
      const lastArr = appendOutput(this, last);
      const tailPieces = [];
      const appendTail = (piece) => {
        const arr = appendOutput(this, piece);
        if (arr.length) tailPieces.push(arr);
      };

      if (typeof b.drain === 'function') {
        this.tailCalls.push('drain');
        // 不带参数时尽量一次取完；若后端按分区返回，循环到空数组为止。
        for (let i = 0; i < 100000; i++) {
          const piece = b.drain(Infinity);
          const arr = toArray(piece);
          if (!arr.length) break;
          appendTail(arr);
        }
      } else if (typeof b.drainAll === 'function') {
        appendTail(b.drainAll());
      } else if (typeof b.readTail === 'function') {
        for (let i = 0; i < 100000; i++) {
          const arr = toArray(b.readTail());
          if (!arr.length) break;
          appendTail(arr);
        }
      } else {
        appendTail(this._callTerminal(['flush', 'finish', 'end']));
      }
      this.tailBuffer = concatArrays(tailPieces);
      this.tailReadOffset = 0;
      return lastArr;
    }

    const out = this._callTerminal(['flush', 'finish', 'end'], chunk);
    this.tailBuffer = new Float64Array(0);
    return appendOutput(this, out);
  }

  _readTail(maxSamples = Infinity) {
    if (this.tailReadOffset >= this.tailBuffer.length) return new Float64Array(0);
    const remainingLength = this.tailBuffer.length - this.tailReadOffset;
    if (maxSamples === Infinity || maxSamples === 'all' || maxSamples >= remainingLength) {
      const out = this.tailBuffer.slice(this.tailReadOffset);
      this.tailReadOffset = this.tailBuffer.length;
      return out;
    }
    const n = Math.max(1, Math.min(maxSamples | 0, remainingLength));
    const out = this.tailBuffer.slice(this.tailReadOffset, this.tailReadOffset + n);
    this.tailReadOffset += n;
    return out;
  }

  drain(maxSamples = Infinity) {
    if (!this.ended) this.end();
    return this._readTail(maxSamples);
  }

  readTail(maxSamples = Infinity) { return this.drain(maxSamples); }
  drainAll() { return this.drain(Infinity); }

  flush(chunk) { return this.end(chunk); }
  finish(chunk) { return this.end(chunk); }

  output() { return concatArrays(this.parts); }

  processAll(x) {
    this.push(x);
    this.end();
    return this.output();
  }
}

export function adaptConvolutionBackend(backend, options = {}) {
  return new FirBackendAdapter(backend, options);
}
export const adaptBackend = adaptConvolutionBackend;
export const createBackendAdapter = adaptConvolutionBackend;
export const ConvolutionBackendAdapter = FirBackendAdapter;
export const BackendAdapter = FirBackendAdapter;
export const TailCompatibleConvolver = FirBackendAdapter;

export function processWithFirBackend(x, kernel, options = {}) {
  const Convention = options.merged
    ? MergedFinalBlockConvolutionBackend
    : options.deferred !== false
      ? DeferredTailConvolutionBackend
      : options.BackendClass || OverlapAddConvolver;
  const backend = options.backend || new Convention(kernel, options);
  const adapter = new FirBackendAdapter(backend, { kernel, ...options });
  const output = adapter.processAll(x);
  return { output, adapter, latency: adapter.latency, tolerance: adapter.tolerance };
}

// 不裁切、不平移的误差统计。长度不一致本身就是失败，避免用文件裁切掩盖延迟变化。
export function convolutionErrorMetrics(actual, expected, options = {}) {
  const a = toArray(actual);
  const e = toArray(expected);
  const compareLength = options.compareLength || Math.min(a.length, e.length);
  let maxAbs = 0;
  let sumSq = 0;
  let energy = 0;
  let mismatchCount = 0;
  for (let i = 0; i < compareLength; i++) {
    const delta = a[i] - e[i];
    const abs = Math.abs(delta);
    if (abs > maxAbs) maxAbs = abs;
    if (abs > (options.zeroEpsilon || 0)) mismatchCount++;
    sumSq += delta * delta;
    energy += e[i] * e[i];
  }
  const rms = Math.sqrt(sumSq / Math.max(1, compareLength));
  const expectedRms = Math.sqrt(energy / Math.max(1, compareLength));
  const relativeRms = rms / (expectedRms || 1);
  const tolerance = { ...TOLERANCE_PROFILES.float64, ...(options.tolerance || {}) };
  const lengthMatch = a.length === e.length;
  const pass = lengthMatch &&
    maxAbs <= tolerance.maxAbs &&
    rms <= tolerance.rms &&
    relativeRms <= (tolerance.relativeRms ?? tolerance.rms);
  return {
    pass,
    precision: tolerance.precision,
    lengthExpected: e.length,
    lengthActual: a.length,
    lengthMatch,
    missingSamples: Math.max(0, e.length - a.length),
    extraSamples: Math.max(0, a.length - e.length),
    compareLength,
    maxAbs,
    rms,
    expectedRms,
    relativeRms,
    mismatchCount,
    tolerance,
  };
}

export function phaseCoherence(actual, reference, options = {}) {
  const a = toArray(actual);
  const r = toArray(reference);
  const n = Math.min(a.length, r.length, options.maxSamples || Infinity);
  let dot = 0, aa = 0, rr = 0;
  for (let i = 0; i < n; i++) {
    dot += a[i] * r[i];
    aa += a[i] * a[i];
    rr += r[i] * r[i];
  }
  const coherence = dot / (Math.sqrt(aa * rr) || 1);
  return {
    coherence,
    phaseRadians: Math.acos(Math.max(-1, Math.min(1, coherence))),
    polarity: coherence < 0 ? -1 : 1,
    pass: Math.abs(coherence) >= (options.threshold ?? TOLERANCE_PROFILES.float64.phaseCoherence) && coherence >= 0,
  };
}

// 对线性相位核，优先使用构造契约；实测峰只作为执行证据。若实测漂移，则
// 明确返回 remeasured，而不是悄悄裁切或移动交付文件。
export function resolveGroupDelayContract(kernel, measured = {}, options = {}) {
  const N = kernel?.length || options.kernelLength || 0;
  const nominal = options.groupDelaySamples ?? (N ? (N - 1) / 2 : null);
  const measuredSamples = Number.isFinite(measured.samples) ? measured.samples : nominal;
  const tolerance = options.peakTolerance ?? 0.75;
  const matches = nominal == null ? true : Math.abs(measuredSamples - nominal) <= tolerance;
  const policy = options.policy || 'auto';
  let timebase;
  let groupDelaySamples;
  if (policy === 'preserve') {
    timebase = 'preserve-legacy-timeline';
    groupDelaySamples = nominal;
  } else if (policy === 'remeasure') {
    timebase = 'remeasured-group-delay';
    groupDelaySamples = measuredSamples;
  } else if (matches) {
    timebase = 'preserve-nominal-group-delay';
    groupDelaySamples = nominal;
  } else {
    timebase = 'remeasured-group-delay';
    groupDelaySamples = measuredSamples;
  }
  return {
    nominalGroupDelaySamples: nominal,
    measuredGroupDelaySamples: measuredSamples,
    measuredSubSample: measured.subSample ?? null,
    matchesNominal: matches,
    timebase,
    groupDelaySamples,
    trimSamples: 0,
    cropSamples: 0,
    algorithmicDelayHiddenByCropping: false,
  };
}

export function measureImpulseAnchor(signal, options = {}) {
  const y = toArray(signal);
  const inputPosition = options.inputPosition || 0;
  const searchFrom = options.searchFrom || 0;
  const searchTo = options.searchTo ?? y.length - 1;
  const peak = findPeak(y, searchFrom, searchTo);
  const samples = peak.index - inputPosition;
  const subSample = peak.subSampleIndex - inputPosition;
  return {
    inputPosition,
    peakIndex: peak.index,
    samples,
    subSample,
    amplitude: peak.amplitude,
    absAmplitude: peak.absAmplitude,
    phasePolarity: peak.amplitude < 0 ? -1 : 1,
  };
}

export function compareImpulsePositionAndPhase(actual, expectedDirect, options = {}) {
  const inputPosition = options.inputPosition || 0;
  const a = measureImpulseAnchor(actual, options);
  const e = measureImpulseAnchor(expectedDirect, options);
  const err = convolutionErrorMetrics(actual, expectedDirect, options);
  const phase = phaseCoherence(actual, expectedDirect, options);
  const peakDelta = a.peakIndex - e.peakIndex;
  const nominal = options.groupDelaySamples;
  const nominalOk = nominal == null ? true : Math.abs(a.samples - nominal) <= (options.peakTolerance ?? 0.75);
  return {
    pass: err.pass && phase.pass && peakDelta === 0 && nominalOk,
    actual: a,
    direct: e,
    peakDeltaSamples: peakDelta,
    nominalGroupDelaySamples: nominal ?? null,
    nominalOk,
    error: err,
    phase,
  };
}

// 原轨是未处理素材时，线性相位/单位延迟核可按群延迟接回原轨做差。
// 返回与原轨等长的对齐差，以及未对齐的完整交付差缓冲。
export function differenceToOriginalTrack(delivered, original, groupDelaySamples = 0) {
  const y = toArray(delivered);
  const x = toArray(original);
  const gd = Math.max(0, Math.round(groupDelaySamples));
  const alignedDifference = new Float64Array(x.length);
  let maxAbs = 0;
  let sumSq = 0;
  for (let i = 0; i < x.length; i++) {
    const v = (y[i + gd] || 0) - x[i];
    alignedDifference[i] = v;
    maxAbs = Math.max(maxAbs, Math.abs(v));
    sumSq += v * v;
  }
  const rawDifference = new Float64Array(y.length);
  for (let i = 0; i < y.length; i++) rawDifference[i] = y[i] - (x[i] || 0);
  return {
    alignedDifference,
    rawDifference,
    alignedMaxAbs: maxAbs,
    alignedRms: Math.sqrt(sumSq / x.length),
    alignmentSamples: gd,
    trimSamples: 0,
  };
}

// 稀疏脉冲尾拍检查：输入中的每个强脉冲只应在输出中产生一个对应主峰。
export function inspectTailBeats(output, input, options = {}) {
  const y = toArray(output);
  const x = toArray(input);
  const gd = options.groupDelaySamples || 0;
  const threshold = options.threshold ?? 0.5;
  const inputPeaks = [];
  for (let i = 0; i < x.length; i++) if (Math.abs(x[i]) >= threshold) inputPeaks.push(i);

  const outputPeaks = [];
  const minDistance = options.minDistance || Math.max(1, Math.round(gd / 2));
  for (let i = 0; i < y.length; i++) {
    if (Math.abs(y[i]) < threshold) continue;
    let isMax = true;
    for (let j = Math.max(0, i - minDistance); j < Math.min(y.length, i + minDistance + 1); j++) {
      if (Math.abs(y[j]) > Math.abs(y[i])) { isMax = false; break; }
    }
    if (isMax && (!outputPeaks.length || i - outputPeaks.at(-1) >= minDistance)) outputPeaks.push(i);
  }

  const matched = [];
  const extra = [];
  for (const p of outputPeaks) {
    const correspondingInput = p - gd;
    if (inputPeaks.some((q) => Math.abs(q - correspondingInput) <= (options.peakTolerance ?? 1))) matched.push(p);
    else extra.push(p);
  }
  const duplicated = extra.length > 0 || outputPeaks.length > inputPeaks.length;
  return {
    inputPeaks,
    outputPeaks,
    matchedPeaks: matched,
    extraPeaks: extra,
    expectedBeatCount: inputPeaks.length,
    actualBeatCount: outputPeaks.length,
    duplicated,
    pass: !duplicated && matched.length === inputPeaks.length,
  };
}

// 末块反相抵消探针：默认使用中心单位脉冲核，使“尾拍是否两次”无歧义。
export function createCancellationTailProbe(options = {}) {
  const blockSize = options.blockSize || 128;
  const kernelLength = options.kernelLength || 65;
  const gd = (kernelLength - 1) / 2;
  const blocks = options.blocks || 4;
  const length = options.length || blockSize * blocks;
  const position = options.position ?? (length - Math.floor(blockSize / 3));
  const lag = options.lag ?? gd + 1;
  const input = new Float64Array(length);
  input[position] = options.amplitude ?? 1;
  // 反相探针与主脉冲隔开 > gd/2：既保留“可反相抵消”的极性证据，也能被尾拍计数分别识别。
  if (options.includeInverse !== false && position + lag < length) input[position + lag] = -(options.amplitude ?? 1);
  const kernel = new Float64Array(kernelLength);
  kernel[gd] = 1;
  return { input, kernel, groupDelaySamples: gd, position, lag, blockSize };
}

export function convolveSwitchedDirect(x, switches = []) {
  const input = toArray(x);
  if (!switches.length) throw new Error('至少需要一个核段');
  const events = switches
    .map((s) => ({ sample: Math.max(0, Math.min(input.length, Math.round(s.sample ?? s.atSample ?? 0))), kernel: toArray(s.kernel) }))
    .sort((a, b) => a.sample - b.sample);
  if (events[0].sample !== 0) events.unshift({ sample: 0, kernel: events[0].kernel });
  const maxN = Math.max(...events.map((e) => e.kernel.length));
  const out = new Float64Array(input.length + maxN - 1);
  for (let i = 0; i < events.length; i++) {
    const start = events[i].sample;
    const end = i + 1 < events.length ? events[i + 1].sample : input.length;
    const segment = input.subarray(start, end);
    const conv = convolveDirect(segment, events[i].kernel);
    for (let j = 0; j < conv.length; j++) out[start + j] += conv[j];
  }
  return out;
}

export function renderSwitchedWithBackend(x, switches = [], options = {}) {
  const input = toArray(x);
  const events = switches
    .map((s) => ({ sample: Math.max(0, Math.min(input.length, Math.round(s.sample ?? s.atSample ?? 0))), kernel: toArray(s.kernel) }))
    .sort((a, b) => a.sample - b.sample);
  if (!events.length) throw new Error('至少需要一个核段');
  if (events[0].sample !== 0) events.unshift({ sample: 0, kernel: events[0].kernel });
  const maxN = Math.max(...events.map((e) => e.kernel.length));
  const out = new Float64Array(input.length + maxN - 1);
  const segmentEvidence = [];
  for (let i = 0; i < events.length; i++) {
    const start = events[i].sample;
    const end = i + 1 < events.length ? events[i + 1].sample : input.length;
    const segment = input.subarray(start, end);
    const Convention = options.tailConvention === 'merged-final'
      ? MergedFinalBlockConvolutionBackend
      : options.tailConvention === 'deferred-drain'
        ? DeferredTailConvolutionBackend
        : (options.BackendClass || OverlapAddConvolver);
    const rendered = new FirBackendAdapter(new Convention(events[i].kernel, options), {
      kernel: events[i].kernel,
      ...options,
    }).processAll(segment);
    for (let j = 0; j < rendered.length; j++) out[start + j] += rendered[j];
    segmentEvidence.push({ start, end, length: segment.length, renderedLength: rendered.length });
  }
  return { output: out, segments: segmentEvidence };
}

export class HotSwapConvolutionRecorder {
  constructor(kernel, options = {}) {
    this.options = options;
    this.blockSize = options.blockSize || 128;
    this.chunks = [];
    this.inputLength = 0;
    this.switches = [{ sample: 0, kernel: toArray(kernel) }];
  }

  push(chunk) {
    const x = toArray(chunk);
    this.chunks.push(x);
    this.inputLength += x.length;
    return this.inputLength;
  }

  write(chunk) { return this.push(chunk); }

  replaceKernel(kernel, atSample) {
    const sample = atSample == null ? this.inputLength : Math.round(atSample);
    this.switches.push({ sample, kernel: toArray(kernel) });
    return sample;
  }

  setKernel(kernel, atSample) { return this.replaceKernel(kernel, atSample); }

  finish(options = {}) {
    const x = concatArrays(this.chunks);
    const expected = convolveSwitchedDirect(x, this.switches);
    const rendered = renderSwitchedWithBackend(x, this.switches, { ...this.options, ...options });
    const error = convolutionErrorMetrics(rendered.output, expected, options);
    return { input: x, expected, ...rendered, error, switches: this.switches };
  }
}

export const LiveKernelSwitchConvolver = HotSwapConvolutionRecorder;

export function migrateFilterConfig(raw) {
  const source = raw && typeof raw === 'object' ? raw : {};
  const oldParams = source.params && typeof source.params === 'object' ? source.params : source;
  const oldSession = source.session && typeof source.session === 'object' ? source.session : {};
  const fromVersion = Number(source.version ?? source.schemaVersion ?? (oldParams.version || 1)) || 1;
  const params = {
    points: Array.isArray(oldParams.points) ? oldParams.points.map((p) => ({ ...p })) : undefined,
    N: oldParams.N ?? 255,
    firType: oldParams.firType || 'I',
    window: oldParams.window || 'hann',
    stopbandDb: oldParams.stopbandDb ?? 70,
    kaiserBeta: oldParams.kaiserBeta,
    sampleRate: oldParams.sampleRate || 48000,
    convolutionBackend: oldParams.convolutionBackend || CONVOLUTION_BACKEND_VERSION,
    tailConvention: oldParams.tailConvention || 'auto',
    latencyPolicy: oldParams.latencyPolicy || 'measure-and-preserve',
    toleranceProfile: oldParams.toleranceProfile || 'float64',
  };
  for (const key of Object.keys(oldParams)) {
    if (!(key in params)) params[key] = oldParams[key];
  }
  if (!params.points) delete params.points;

  const session = { ...oldSession };
  const migrated = fromVersion < CONFIG_SCHEMA_VERSION;
  return {
    app: source.app || 'linear-phase-eq',
    version: CONFIG_SCHEMA_VERSION,
    schemaVersion: CONFIG_SCHEMA_VERSION,
    params,
    session,
    migration: {
      migrated,
      fromVersion,
      toVersion: CONFIG_SCHEMA_VERSION,
      backendVersion: CONVOLUTION_BACKEND_VERSION,
      preservedLegacyTimeline: true,
      cropSamples: 0,
      migratedAt: new Date(0).toISOString().replace('1970-01-01T00:00:00.000Z', 'deterministic'),
    },
  };
}

// 旧 JSON 载入后立即真实卷积一段长音频；不读取 output/ 下的手工产物。
export function loadMigratedConfigAndProcess(raw, source, options = {}) {
  const config = migrateFilterConfig(raw);
  const design = designFir({
    points: config.params.points,
    N: config.params.N,
    firType: config.params.firType,
    window: config.params.window,
    kaiserBeta: config.params.kaiserBeta,
    stopbandDb: config.params.stopbandDb,
    sampleRate: config.params.sampleRate,
  });
  if (!design.ok) return { ok: false, config, design };
  const length = options.longAudioSamples || config.params.sampleRate * 4;
  const x = source || Float64Array.from({ length }, (_, i) =>
    0.25 * Math.sin(2 * Math.PI * 55 * i / config.params.sampleRate) +
    0.12 * Math.sin(i * 0.117) +
    0.05 * (Math.sin(i * 12.9898) * 43758.5453 % 1)
  );
  const result = processWithFirBackend(x, design.kernel, {
    blockSize: options.blockSize || 128,
    precision: config.params.toleranceProfile,
  });
  const expected = convolveDirect(x, design.kernel);
  const error = convolutionErrorMetrics(result.output, expected, { tolerance: TOLERANCE_PROFILES[config.params.toleranceProfile] });
  return { ok: error.pass, config, design, input: x, output: result.output, expected, error, latency: result.latency };
}

function deterministicSignal(n) {
  // 不依赖手工数据：多正弦 + 确定性伪随机，覆盖低频、高频和非整除块。
  return Float64Array.from({ length: n }, (_, i) =>
    0.35 * Math.sin(i * 0.0713) +
    0.22 * Math.sin(i * 0.0179 + 1.7) +
    0.12 * Math.cos(i * 0.217) +
    0.04 * Math.sin(i * 1.713)
  );
}

export function verifyFirBackend(options = {}) {
  const N = options.N || (options.long ? 2049 : 255);
  const inputLength = options.inputLength || (options.long ? 8192 : 2048);
  const blockSize = options.blockSize || 128;
  const points = options.points || [
    { f: 0, db: 0 }, { f: 0.18, db: -0.5 }, { f: 0.42, db: -9 }, { f: 1, db: -42 },
  ];
  const design = designFir({ points, N, firType: 'I', window: 'hann', sampleRate: 48000 });
  const checks = [];
  if (!design.ok) {
    return { ok: false, pass: false, backendVersion: CONVOLUTION_BACKEND_VERSION, checks, message: design.message };
  }
  const h = design.kernel;
  const gd = design.groupDelaySamples;
  const x = deterministicSignal(inputLength);
  const direct = convolveDirect(x, h);

  for (const [name, BackendClass] of [
    ['deferred-drain', DeferredTailConvolutionBackend],
    ['merged-final', MergedFinalBlockConvolutionBackend],
  ]) {
    const backend = new BackendClass(h, { blockSize });
    const adapter = new FirBackendAdapter(backend, { kernel: h, blockSize });
    // 用非规则切片推送，覆盖长滤波器、块边界和输入结束。
    let pos = 0;
    let step = 1;
    while (pos < x.length) {
      const n = Math.min(x.length - pos, [1, 7, blockSize - 1, blockSize, blockSize + 13, 311][step % 6]);
      adapter.push(x.subarray(pos, pos + n));
      pos += n;
      step++;
    }
    adapter.end();
    const actual = adapter.output();
    const error = convolutionErrorMetrics(actual, direct);
    checks.push({ id: `convolution-${name}`, backend: name, pass: error.pass, ...error });
  }

  const impulsePosition = inputLength - blockSize + 17;
  const impulseInput = new Float64Array(inputLength);
  impulseInput[impulsePosition] = 1;
  const impulseDirect = convolveDirect(impulseInput, h);
  for (const [name, BackendClass] of [
    ['deferred-drain', DeferredTailConvolutionBackend],
    ['merged-final', MergedFinalBlockConvolutionBackend],
  ]) {
    const actual = new BackendClass(h, { blockSize }).processAll(impulseInput);
    const anchor = compareImpulsePositionAndPhase(actual, impulseDirect, {
      inputPosition: 0,
      groupDelaySamples: impulsePosition + gd,
      searchFrom: impulsePosition,
      searchTo: Math.min(actual.length - 1, impulsePosition + N + 32),
    });
    const contract = resolveGroupDelayContract(h, { samples: anchor.actual.samples - impulsePosition, subSample: anchor.actual.subSample - impulsePosition });
    checks.push({ id: `impulse-phase-${name}`, backend: name, pass: anchor.pass && contract.trimSamples === 0, anchor, contract });
  }

  const probe = createCancellationTailProbe({ blockSize, kernelLength: options.probeKernelLength || 65 });
  for (const [name, BackendClass] of [
    ['deferred-drain', DeferredTailConvolutionBackend],
    ['merged-final', MergedFinalBlockConvolutionBackend],
  ]) {
    const actual = new BackendClass(probe.kernel, { blockSize }).processAll(probe.input);
    const beats = inspectTailBeats(actual, probe.input, { groupDelaySamples: probe.groupDelaySamples, threshold: 0.5 });
    const originalDifference = differenceToOriginalTrack(actual, probe.input, probe.groupDelaySamples);
    const pass = beats.pass && originalDifference.alignedMaxAbs < 1e-8;
    checks.push({ id: `tail-once-${name}`, backend: name, pass, beats, originalDifference });
  }

  const switchSample = Math.floor(inputLength * 0.37);
  const h2Design = designFir({ points, N: N > 513 ? 513 : 127, firType: 'I', window: 'hann' });
  if (h2Design.ok) {
    const switchedExpected = convolveSwitchedDirect(x, [
      { sample: 0, kernel: h },
      { sample: switchSample, kernel: h2Design.kernel },
    ]);
    for (const convention of ['deferred-drain', 'merged-final']) {
      const rendered = renderSwitchedWithBackend(x, [
        { sample: 0, kernel: h },
        { sample: switchSample, kernel: h2Design.kernel },
      ], { blockSize, tailConvention: convention });
      const error = convolutionErrorMetrics(rendered.output, switchedExpected);
      checks.push({ id: `hot-switch-${convention}`, backend: convention, pass: error.pass, switchSample, ...error });
    }
  }

  const migration = migrateFilterConfig({ app: 'linear-phase-eq', version: 1, params: { N: 333, window: 'hann' } });
  checks.push({
    id: 'config-v1-migration',
    pass: migration.version === 2 && migration.params.N === 333 && migration.migration.cropSamples === 0,
    migration,
  });

  const pass = checks.every((c) => c.pass);
  return {
    ok: pass,
    pass,
    long: !!options.long,
    backendVersion: CONVOLUTION_BACKEND_VERSION,
    apiVersion: FIR_BACKEND_API_VERSION,
    buildLock: BACKEND_BUILD_LOCK,
    tolerance: TOLERANCE_PROFILES.float64,
    groupDelaySamples: gd,
    convolutionError: checks.filter((c) => c.id.startsWith('convolution-')).map((c) => ({
      backend: c.backend, maxAbs: c.maxAbs, rms: c.rms, relativeRms: c.relativeRms,
      lengthActual: c.lengthActual, lengthExpected: c.lengthExpected,
    })),
    checks,
  };
}

export function quickFirBackendEvidence(options = {}) {
  return verifyFirBackend({ N: 127, inputLength: 1024, blockSize: 128, probeKernelLength: 33, ...options });
}

export const BACKEND_VERSION = CONVOLUTION_BACKEND_VERSION;
export const FIR_BACKEND_VERSION = FIR_BACKEND_API_VERSION;
export const TOLERANCES = TOLERANCE_PROFILES;
export const TOLERANCE = TOLERANCE_PROFILES;
export const DEFAULT_TOLERANCE = TOLERANCE_PROFILES.float64;
export const migrateConfig = migrateFilterConfig;
export const verifyBackend = verifyFirBackend;
export const runBackendCheck = verifyFirBackend;
export const createTailProbe = createCancellationTailProbe;
export const differenceToSourceTrack = differenceToOriginalTrack;
export const createFirBackend = adaptConvolutionBackend;
export const createBackend = adaptConvolutionBackend;
export const adaptFirBackend = adaptConvolutionBackend;
export const FirBackend = FirBackendAdapter;
export const FIRBackend = FirBackendAdapter;
export const FIRBackendAdapter = FirBackendAdapter;
export const DeferredDrainBackend = DeferredTailConvolutionBackend;
export const DeferredTailBackend = DeferredTailConvolutionBackend;
export const LongPartitionDrainBackend = DeferredTailConvolutionBackend;
export const MergedFinalBackend = MergedFinalBlockConvolutionBackend;
export const MergedFinalConvolutionBackend = MergedFinalBlockConvolutionBackend;
export function createConvolutionBackend(type, kernel, options = {}) {
  const kind = normalizeTailConvention(type || options.tailConvention, 'deferred-drain');
  const BackendClass = kind === 'merged-final'
    ? MergedFinalBlockConvolutionBackend
    : DeferredTailConvolutionBackend;
  return new BackendClass(kernel, options);
}
export const createTailConvolutionBackend = createConvolutionBackend;
export const TailFlushAdapter = FirBackendAdapter;
export function groupDelaySamples(kernel) {
  const N = typeof kernel === 'number' ? kernel : kernel?.length ?? 0;
  return N ? (N - 1) / 2 : 0;
}
export function expectedConvolutionLength(input, kernel) {
  return Math.max(0, (input?.length ?? 0) + (kernel?.length ?? 0) - 1);
}
export class FirConvolutionBackend extends DeferredTailConvolutionBackend {}
export class ConvolutionBackend extends DeferredTailConvolutionBackend {}

export default {
  FIR_BACKEND_API_VERSION,
  CONVOLUTION_BACKEND_VERSION,
  CONFIG_SCHEMA_VERSION,
  BACKEND_BUILD_LOCK,
  TOLERANCE_PROFILES,
  FirBackendAdapter,
  DeferredTailConvolutionBackend,
  MergedFinalBlockConvolutionBackend,
  HotSwapConvolutionRecorder,
  adaptConvolutionBackend,
  processWithFirBackend,
  convolveSwitchedDirect,
  renderSwitchedWithBackend,
  migrateFilterConfig,
  verifyFirBackend,
  quickFirBackendEvidence,
};
