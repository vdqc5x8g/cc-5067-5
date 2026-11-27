// 高层管线：参数 -> 线性相位核 -> 真实卷积音频。界面导出与 HTTP 渲染共用。

import { designFir, scanLengthTradeoff } from './fir-design.js';
import { convolveDirect } from './convolution.js';
import {
  processWithFirBackend,
  CONVOLUTION_BACKEND_VERSION,
  TOLERANCE_PROFILES,
} from './fir-backend.js';
import {
  generateImpulse,
  generateSweep,
  generateDrumLoop,
} from './audio.js';
import { defaultTargetPoints } from './target.js';

export function designFromParams(params = {}) {
  const sampleRate = params.sampleRate || 48000;
  const points = params.points || defaultTargetPoints();
  return designFir({
    points,
    N: params.N || 255,
    firType: params.firType || 'I',
    window: params.window || 'hann',
    kaiserBeta: params.kaiserBeta,
    stopbandDb: params.stopbandDb,
    sampleRate,
    cutoffNyq: params.cutoffNyq,
    specs: params.specs,
  });
}

export function generateMaterial(material, sampleRate = 48000, opts = {}) {
  switch (material) {
    case 'impulse':
      return generateImpulse({ sampleRate, length: opts.length || sampleRate, position: 0 });
    case 'sweep':
      return generateSweep({ sampleRate, durationSec: opts.durationSec || 3 });
    case 'drums':
      return generateDrumLoop({ sampleRate, bpm: opts.bpm || 124, bars: opts.bars || 2 });
    case 'silence':
      return new Float64Array(sampleRate);
    default:
      return generateDrumLoop({ sampleRate, bpm: opts.bpm || 124, bars: 2 });
  }
}

// 离线处理单声道素材，返回核设计与包含完整尾部的输出。
export function processMaterial(params = {}, source) {
  const design = designFromParams(params);
  if (!design.ok) return { ok: false, design };
  const x = source || generateMaterial(params.material || 'drums', design.sampleRate, params);
  const kernel = Float64Array.from(design.kernel);
  const useStreaming = params.streaming !== false;
  const precision = params.precision || 'float64';
  let y;
  let backendEvidence = null;
  if (useStreaming) {
    const result = processWithFirBackend(x, kernel, {
      blockSize: params.blockSize || 128,
      tailConvention: params.tailConvention || 'deferred-drain',
      precision,
    });
    y = result.output;
    backendEvidence = {
      version: CONVOLUTION_BACKEND_VERSION,
      adapter: 'FirBackendAdapter',
      tailConvention: result.adapter.tailConvention,
      latency: result.latency,
      tolerance: TOLERANCE_PROFILES[precision] || TOLERANCE_PROFILES.float64,
    };
  } else {
    y = convolveDirect(x, kernel);
  }
  return {
    ok: true,
    design,
    input: x,
    output: y,
    sampleRate: design.sampleRate,
    backend: backendEvidence,
  };
}

export { scanLengthTradeoff };
