// 客观自检：这些检查同时服务于界面“内核校验”面板、HTTP /api/verify 与 npm test。
// 任何“只画理想曲线却播放未处理声音 / 只移动显示时标”的实现都过不了这里的测量。

import { designFir } from './fir-design.js';
import { convolveDirect, streamConvolve } from './convolution.js';
import { runSessionSync } from './session.js';
import { preRingingMetrics, generateImpulse } from './audio.js';
import { quickFirBackendEvidence, CONVOLUTION_BACKEND_VERSION } from './fir-backend.js';

function check(id, title, pass, detail) {
  return { id, title, pass: !!pass, detail: detail == null ? '' : String(detail) };
}

export function runVerification() {
  const results = [];
  const pts = [
    { f: 0, db: 4.5 },
    { f: 0.08, db: 4.5 },
    { f: 0.22, db: 1.2 },
    { f: 0.45, db: 0 },
    { f: 0.7, db: -1.5 },
    { f: 1, db: -1.5 },
  ];

  // 1. I 型对称性
  const d1 = designFir({ points: pts, N: 127, window: 'hann' });
  results.push(
    check(
      'sym1',
      'I 型核严格偶对称 h[n]=h[N-1-n]',
      d1.ok && d1.symmetryResidual.symmetricMaxError < 1e-10,
      d1.ok ? `最大对称残差 ${d1.symmetryResidual.symmetricMaxError.toExponential(2)}` : d1.message
    )
  );

  // 2. II 型对称性 + Nyquist 失败反馈
  const d2 = designFir({ points: [{ f: 0, db: 0 }, { f: 1, db: -60 }], N: 128, firType: 'II' });
  results.push(
    check(
      'sym2',
      'II 型偶数长核保持偶对称',
      d2.ok && d2.symmetryResidual.symmetricMaxError < 1e-10,
      d2.ok ? `残差 ${d2.symmetryResidual.symmetricMaxError.toExponential(2)}` : d2.message
    )
  );
  const d2bad = designFir({ points: [{ f: 0, db: 0 }, { f: 1, db: 0 }], N: 128, firType: 'II' });
  results.push(
    check(
      'type2fail',
      'II 型无法实现 Nyquist 增益时明确失败并给建议',
      !d2bad.ok && d2bad.error === 'TYPE2_NYQUIST_ZERO' && d2bad.suggestion?.firType === 'I',
      d2bad.ok ? '错误地成功了' : d2bad.message
    )
  );

  // 3. 群延迟恒定
  results.push(
    check(
      'gd',
      '实测群延迟恒等于 (N-1)/2 = 63 样本',
      d1.ok &&
        Math.abs(d1.groupDelay.measuredMean - 63) < 1e-6 &&
        d1.groupDelay.maxDeviationSamples < 1e-6,
      d1.ok
        ? `均值 ${d1.groupDelay.measuredMean.toFixed(6)}，最大偏离 ${d1.groupDelay.maxDeviationSamples.toExponential(2)}`
        : ''
    )
  );

  // 4. 目标逼近
  const d3 = designFir({ points: pts, N: 255, window: 'kaiser', stopbandDb: 70 });
  results.push(
    check(
      'approx',
      '实际幅度响应对目标的最大误差 < 1.5 dB',
      d3.ok && d3.approximation.maxErrorDb < 1.5,
      d3.ok ? `最大 ${d3.approximation.maxErrorDb.toFixed(3)} dB，RMS ${d3.approximation.rmsErrorDb.toFixed(4)} dB` : ''
    )
  );

  // 5. 直接卷积已知向量
  const known = convolveDirect(Float64Array.from([1, 2, 3]), Float64Array.from([1, 1]));
  const knownOk = known.length === 4 && [1, 3, 5, 3].every((v, i) => Math.abs(v - known[i]) < 1e-12);
  results.push(check('direct', '直接卷积已知结果 [1,2,3]*[1,1]=[1,3,5,3]', knownOk, Array.from(known).join(',')));

  // 6. 流式 == 直接（多块长，含非整除尾部）
  let maxErr = 0;
  let lenOk = true;
  for (const B of [1, 3, 17, 64, 128, 257, 1000]) {
    const x = Float64Array.from({ length: 2000 }, (_, i) => Math.sin(i * 0.071) + 0.3 * Math.cos(i * 0.013));
    const ref = convolveDirect(x, d1.kernel);
    const got = streamConvolve(x, d1.kernel, B);
    if (got.length !== ref.length) lenOk = false;
    for (let i = 0; i < Math.min(ref.length, got.length); i++) maxErr = Math.max(maxErr, Math.abs(ref[i] - got[i]));
  }
  results.push(
    check(
      'stream',
      '分块 overlap-add（含重叠区与尾部拼接）与直接卷积一致，长度=x+N-1',
      lenOk && maxErr < 1e-9,
      `7 种块长最大误差 ${maxErr.toExponential(2)}，长度${lenOk ? '正确' : '错误'}`
    )
  );

  // 7. 脉冲实际峰位 = 算法延迟，且前振铃可测
  const imp = generateImpulse({ length: 4096, position: 0 });
  const out = convolveDirect(imp, d1.kernel);
  let peakIdx = 0;
  for (let i = 1; i < d1.N; i++) if (Math.abs(out[i]) > Math.abs(out[peakIdx])) peakIdx = i;
  const pre = preRingingMetrics(d1.kernel, 63);
  results.push(
    check(
      'latency',
      '脉冲经真实卷积后的峰位 = 63 样本（算法延迟，未在播放起点隐藏）',
      peakIdx === 63,
      `实测峰位 ${peakIdx}；前振铃能量 ${pre.preEnergyDb.toFixed(1)} dB，峰前最大旁瓣 ${pre.preToPeakDb.toFixed(1)} dB`
    )
  );

  // 8. 输出尾部完整保留
  results.push(
    check(
      'tail',
      '卷积输出长度 = x.length + N - 1，尾部 N-1 样本保留',
      out.length === imp.length + d1.N - 1,
      `输出 ${out.length} = ${imp.length}+${d1.N}-1`
    )
  );

  // 9. 双路径峰位/绝对拍差：默认配置 old-tail 下 = Δgd
  const sOld = runSessionSync({ scenario: 'old-tail' });
  const delta = sOld.resolved.deltaGd;
  results.push(
    check(
      'peaks-old',
      '旧尾鼓点：回印(deferred) 与耳返(dual-new) 实测峰差 = Δgd',
      sOld.difference.absoluteSamples === delta,
      `回印峰 ${sOld.echoMeasure.peakSampleFile}，耳返峰 ${sOld.iemMeasure.peakSampleFile}，差 ${sOld.difference.absoluteSamples}（期望 ${delta}），拍差 ${sOld.difference.absoluteBeats.toFixed(5)}`
    )
  );
  const sNew = runSessionSync({ scenario: 'new-effective' });
  results.push(
    check(
      'peaks-new',
      '新核生效后的鼓点：两路径峰差 = 0',
      sNew.difference.absoluteSamples === 0,
      `差 ${sNew.difference.absoluteSamples}`
    )
  );

  // 10. 梳状检测：未对齐检出，对齐不检出
  results.push(
    check(
      'comb-raw',
      '未对齐双核相加被检出梳状抵消',
      sOld.comb.raw.detected === true,
      `${sOld.comb.raw.notchCount} 个陷波，最深 ${sOld.comb.raw.deepestDepthDb?.toFixed(1)} dB，间隔 ${sOld.comb.raw.medianSpacingHz?.toFixed(1)} Hz`
    )
  );
  results.push(
    check(
      'comb-aligned',
      '群延迟对齐后不再出现梳状抵消',
      sOld.comb.aligned.detected === false,
      `陷波数 ${sOld.comb.aligned.notchCount}`
    )
  );

  const backend = quickFirBackendEvidence();

  const pass = results.every((r) => r.pass) && backend.pass;
  return {
    ok: pass,
    pass,
    passed: results.filter((r) => r.pass).length,
    total: results.length,
    results,
    convolutionBackendVersion: CONVOLUTION_BACKEND_VERSION,
    backend: {
      pass: backend.pass,
      backendVersion: backend.backendVersion,
      groupDelaySamples: backend.groupDelaySamples,
      convolutionError: backend.convolutionError,
      checks: backend.checks.map((c) => ({ id: c.id, pass: c.pass })),
    },
  };
}
