// 线性相位 EQ 母带面板主控制器。
// 唯一事实源是 state.design.kernel：所有曲线/指标/试听卷积核都从它驱动。

import { designFir, scanLengthTradeoff } from '/dsp/fir-design.js';
import { designFromParams, processMaterial, generateMaterial } from '/dsp/pipeline.js';
import { runSession } from '/dsp/session.js';
import { runVerification } from '/dsp/verify.js';
import { encodeWavBytes } from '/dsp/wav.js';
import { preRingingMetrics } from '/dsp/audio.js';
import { defaultTargetPoints, PRESET_TARGETS, clamp } from '/dsp/target.js';
import { svgEl, clear, setupFrame, pathFromSeries, pathFromPoints, renderStrip, drawWaveform, drawBars, fmtTick } from './plots.js';

const $ = (id) => document.getElementById(id);
const STORE_KEY = 'lpeq-params-v1';

const state = {
  points: defaultTargetPoints(),
  N: 255,
  firType: 'I',
  window: 'hann',
  stopband: 70,
  sampleRate: 48000,
  design: null,
  compare: null,
  dragIndex: -1,
  eqGeom: null,
  auditionMaterial: 'drums',
  uploadedBuffer: null,
  session: null,
  urls: { echo: null, iem: null },
};

/* ---------------- 参数收集 / 持久化 ---------------- */

function collectParams() {
  return {
    points: state.points.map((p) => ({ f: p.f, db: p.db })),
    N: state.N,
    firType: state.firType,
    window: state.window,
    stopbandDb: Number($('input-stopband').value),
    kaiserBeta: undefined,
    sampleRate: Number($('select-sampleRate').value),
  };
}

function collectSessionConfig() {
  return {
    scenario: $('select-scenario').value,
    N1: Number($('input-N1').value),
    N2: Number($('input-N2').value),
    bpm: Number($('input-bpm').value),
    iemBlock: Number($('input-iemBlock').value),
    material: $('select-session-material').value,
    echoMode: $('select-echo-mode').value,
    iemMode: $('select-iem-mode').value,
    targetPoints: state.points,
    window: state.window,
    sampleRate: state.sampleRate,
  };
}

let saveTimer = 0;
function saveLocal() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => {
    try {
      localStorage.setItem(STORE_KEY, JSON.stringify({ params: collectParams(), session: collectSessionConfig() }));
    } catch { /* 隐私模式忽略 */ }
  }, 250);
}

function applyPersisted() {
  let raw = null;
  try { raw = JSON.parse(localStorage.getItem(STORE_KEY) || 'null'); } catch { raw = null; }
  if (raw?.params) applyParams(raw.params, false);
  if (raw?.session) {
    for (const [id, key] of [
      ['select-scenario', 'scenario'], ['input-N1', 'N1'], ['input-N2', 'N2'],
      ['input-bpm', 'bpm'], ['input-iemBlock', 'iemBlock'], ['select-session-material', 'material'],
      ['select-echo-mode', 'echoMode'], ['select-iem-mode', 'iemMode'],
    ]) {
      if (raw.session[key] != null) $(id).value = raw.session[key];
    }
  }
}

function applyParams(p, persist = true) {
  if (!p) return;
  if (Array.isArray(p.points) && p.points.length >= 2) state.points = p.points.map((q) => ({ f: clamp(q.f, 0, 1), db: clamp(q.db, -60, 12) }));
  if (p.N != null) { state.N = Number(p.N); $('input-N').value = state.N; $('val-N').textContent = state.N; }
  if (p.firType) { state.firType = p.firType; $('select-firType').value = p.firType; }
  if (p.window) { state.window = p.window; $('select-window').value = p.window; }
  if (p.stopbandDb != null) { $('input-stopband').value = p.stopbandDb; $('val-stopband').textContent = p.stopbandDb + ' dB'; }
  if (p.sampleRate) { state.sampleRate = Number(p.sampleRate); $('select-sampleRate').value = String(p.sampleRate); }
  updateKaiserVisibility();
  redesign();
  if (persist) saveLocal();
}

/* ---------------- 设计 + 全界面刷新 ---------------- */

function redesign() {
  state.N = Number($('input-N').value);
  state.firType = $('select-firType').value;
  state.window = $('select-window').value;
  state.sampleRate = Number($('select-sampleRate').value);
  const stopbandDb = Number($('input-stopband').value);

  const result = designFir({
    points: state.points,
    N: state.N,
    firType: state.firType,
    window: state.window,
    stopbandDb,
    sampleRate: state.sampleRate,
  });

  const fb = $('kernel-feedback');
  if (!result.ok) {
    fb.className = 'feedback show error';
    fb.innerHTML = `内核设计失败：${result.message}` +
      (result.suggestion?.firType ? ` <button class="btn small" id="btn-fix-type">自动改为 ${result.suggestion.firType} 型</button>` : '');
    const fix = $('btn-fix-type');
    if (fix) fix.onclick = () => {
      $('select-firType').value = result.suggestion.firType;
      redesign();
    };
    // 失败反馈也推给实时卷积：保持上一个有效核，绝不让界面播放与曲线无关的声音
    return;
  }

  state.design = result;
  $('input-N').value = result.N;
  $('val-N').textContent = result.N;
  fb.className = 'feedback';
  fb.textContent = '';

  renderEq();
  renderResponse();
  renderMetrics();
  evaluateSpec();
  postWorkletKernel();
  saveLocal();
}

function evaluateSpec() {
  const maxTr = Number($('spec-transition').value);
  const maxErr = Number($('spec-error').value);
  const badge = $('spec-status');
  if (!state.design) { badge.className = 'badge'; badge.textContent = '无有效内核'; return; }
  const fc = 0.35;
  const bw = designFir({
    points: [
      { f: 0, db: 0 }, { f: fc, db: 0 },
      { f: fc + 1e-4, db: -90 }, { f: 1, db: -90 },
    ],
    N: state.design.N, firType: 'I', window: state.window,
    stopbandDb: Number($('input-stopband').value), sampleRate: state.sampleRate, cutoffNyq: fc,
  });
  const checks = [];
  if (Number.isFinite(maxTr) && bw.ok && bw.cutoffTransition?.available) {
    checks.push({ label: `过渡 ${Math.round(bw.cutoffTransition.widthHz)}Hz≤${maxTr}`, pass: bw.cutoffTransition.widthHz <= maxTr });
  }
  if (Number.isFinite(maxErr) && state.design.approximation?.maxErrorDb != null) {
    checks.push({ label: `误差 ${state.design.approximation.maxErrorDb.toFixed(2)}≤${maxErr}dB`, pass: state.design.approximation.maxErrorDb <= maxErr });
  }
  const allPass = checks.length && checks.every((c) => c.pass);
  badge.className = 'badge ' + (allPass ? 'good' : 'bad');
  badge.textContent = checks.map((c) => (c.pass ? '✓' : '✗') + ' ' + c.label).join('  ') || '—';
}

function renderMetrics() {
  const d = state.design;
  if (!d) return;
  $('metric-gd').textContent = String(d.groupDelaySamples);
  $('metric-latency').textContent = d.latencyMs.toFixed(2);
  $('metric-tail').textContent = d.tailSamples;
  $('metric-sym').textContent = d.symmetryResidual.symmetricMaxError < 1e-12
    ? '<1e-12'
    : d.symmetryResidual.symmetricMaxError.toExponential(1);
  const tr = d.transition;
  $('metric-trans').textContent = tr?.available ? Math.round(tr.widthHz) : '平缓目标';
  $('metric-err').textContent = d.approximation?.maxErrorDb != null ? d.approximation.maxErrorDb.toFixed(2) : '—';
  $('metric-gddev').textContent = d.groupDelay.maxDeviationSamples < 1e-9
    ? '<1e-9'
    : d.groupDelay.maxDeviationSamples.toExponential(1);
  const pre = preRingingMetrics(d.kernel, Math.round(d.groupDelaySamples));
  $('metric-prering').textContent = pre.preEnergyDb.toFixed(1);
}

/* ---------------- EQ 编辑器 ---------------- */

const EQ_W = 760, EQ_H = 340, EQ_M = { l: 48, r: 16, t: 16, b: 30 };
const DB_MIN = -64, DB_MAX = 14;

function eqInvert(ev) {
  const svg = $('eq-svg');
  const rect = svg.getBoundingClientRect();
  const x = ((ev.clientX - rect.left) / rect.width) * EQ_W;
  const y = ((ev.clientY - rect.top) / rect.height) * EQ_H;
  const f = clamp((x - EQ_M.l) / (EQ_W - EQ_M.l - EQ_M.r), 0, 1);
  const db = clamp(DB_MAX - ((y - EQ_M.t) / (EQ_H - EQ_M.t - EQ_M.b)) * (DB_MAX - DB_MIN), -60, 12);
  return { x, y, f, db };
}

function renderEq() {
  const svg = $('eq-svg');
  const sr2 = state.sampleRate / 2;
  const geom = setupFrame(
    svg, EQ_W, EQ_H, EQ_M, [0, 1], [DB_MIN, DB_MAX],
    {
      xTicks: 10, yTicks: 13,
      fmtX: (v) => { const hz = v * sr2; return hz >= 1000 ? (hz / 1000).toFixed(1) + 'k' : Math.round(hz); },
      fmtY: (v) => fmtTick(v),
    }
  );
  state.eqGeom = geom;

  // 0dB 已由 setupFrame 加重；实际幅度曲线
  if (state.design) {
    const mag = state.design.response.magDb;
    const d = pathFromSeries(mag, geom.sx, geom.sy, 0, 1, (v) => clamp(v, DB_MIN - 10, DB_MAX + 4));
    svgEl('path', { d, fill: 'none', stroke: '#4fb4ff', 'stroke-width': 2.2, 'stroke-linejoin': 'round' }, svg);
  }

  // 目标曲线（橙色虚线 + 半透明区域）
  const targetD = pathFromPoints(state.points, geom.sx, geom.sy);
  const areaD = targetD + `L${geom.sx(state.points.at(-1).f)},${geom.sy(DB_MIN)} L${geom.sx(state.points[0].f)},${geom.sy(DB_MIN)} Z`;
  svgEl('path', { d: areaD, fill: 'rgba(255,180,84,0.08)', stroke: 'none' }, svg);
  svgEl('path', { d: targetD, fill: 'none', stroke: '#ffb454', 'stroke-width': 2, 'stroke-dasharray': '7 4' }, svg);

  // 控制点
  state.points.forEach((p, i) => {
    const c = svgEl('circle', {
      cx: geom.sx(p.f), cy: geom.sy(p.db), r: 6.5,
      fill: i === state.dragIndex ? '#fff' : '#ffb454',
      stroke: '#1a1206', 'stroke-width': 2, style: 'cursor:grab', 'data-i': i,
    }, svg);
    c.addEventListener('pointerdown', (e) => startDrag(e, i));
    c.addEventListener('dblclick', (e) => {
      e.stopPropagation();
      if (state.points.length <= 2) return;
      state.points.splice(i, 1);
      redesign();
    });
  });
}

let rafDrag = 0;
function startDrag(ev, index) {
  ev.preventDefault();
  ev.stopPropagation();
  state.dragIndex = index;
  const onMove = (e) => {
    const q = eqInvert(e);
    const pts = state.points;
    const i = index;
    pts[i].db = Math.round(q.db * 10) / 10;
    // 频率保持顺序；首尾频率锁定 0 / 1
    if (i !== 0 && i !== pts.length - 1) {
      const lo = pts[i - 1].f + 0.002;
      const hi = pts[i + 1].f - 0.002;
      pts[i].f = clamp(q.f, Math.min(lo, hi), Math.max(lo, hi));
    }
    cancelAnimationFrame(rafDrag);
    rafDrag = requestAnimationFrame(redesign);
  };
  const onUp = () => {
    state.dragIndex = -1;
    window.removeEventListener('pointermove', onMove);
    window.removeEventListener('pointerup', onUp);
    renderEq();
    saveLocal();
  };
  window.addEventListener('pointermove', onMove);
  window.addEventListener('pointerup', onUp);
}

function initEqInteractions() {
  const svg = $('eq-svg');
  svg.addEventListener('dblclick', (ev) => {
    if (ev.target.tagName === 'circle') return;
    const q = eqInvert(ev);
    if (state.points.length >= 24) return;
    state.points.push({ f: Number(q.f.toFixed(4)), db: Math.round(q.db * 10) / 10 });
    state.points.sort((a, b) => a.f - b.f);
    redesign();
  });

  $('preset-row').addEventListener('click', (e) => {
    const btn = e.target.closest('[data-preset]');
    if (!btn) return;
    const preset = PRESET_TARGETS[btn.dataset.preset];
    if (!preset) return;
    state.points = preset.points.map((p) => ({ ...p }));
    redesign();
  });
}

/* ---------------- 响应三联图 ---------------- */

function renderResponse() {
  const svg = $('response-svg');
  clear(svg);
  const d = state.design;
  if (!d) return;
  const W = 760;
  const showMag = $('resp-mag').checked;
  const showPhase = $('resp-phase').checked;
  const showGd = $('resp-gd').checked;
  const strips = [];
  if (showMag) strips.push({ key: 'mag', h: 210 });
  if (showPhase) strips.push({ key: 'phase', h: 120 });
  if (showGd) strips.push({ key: 'gd', h: 120 });
  if (!strips.length) return;
  const gap = 10;
  const H = strips.reduce((a, s) => a + s.h, 0) + gap * (strips.length - 1);
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  svg.style.height = 'auto';

  let y = 0;
  for (const s of strips) {
    if (s.key === 'mag') drawMagStrip(svg, { x: 0, y, w: W, h: s.h });
    if (s.key === 'phase') drawPhaseStrip(svg, { x: 0, y, w: W, h: s.h });
    if (s.key === 'gd') drawGdStrip(svg, { x: 0, y, w: W, h: s.h });
    y += s.h + gap;
  }
}

function drawMagStrip(svg, r) {
  const frame = renderStrip(svg, r, [0, 1], [-90, 16], {
    title: '幅度 dB（实测 FFT）', fmtX: (v) => {
      const hz = v * state.sampleRate / 2;
      return hz >= 1000 ? (hz / 1000).toFixed(1) + 'k' : Math.round(hz);
    }, fmtY: (v) => fmtTick(v),
  });
  const d = state.design;
  const dd = pathFromSeries(d.response.magDb, frame.sx, frame.sy, 0, 1, (v) => clamp(v, -95, 20));
  svgEl('path', { d: dd, fill: 'none', stroke: '#4fb4ff', 'stroke-width': 2 }, frame.g);
  const td = pathFromPoints(state.points, frame.sx, frame.sy);
  svgEl('path', { d: td, fill: 'none', stroke: '#ffb454', 'stroke-width': 1.6, 'stroke-dasharray': '6 4' }, frame.g);
  if (state.compare) {
    const cd = pathFromSeries(state.compare.response.magDb, frame.sx, frame.sy, 0, 1, (v) => clamp(v, -95, 20));
    svgEl('path', { d: cd, fill: 'none', stroke: '#7d8aa0', 'stroke-width': 1.6, 'stroke-dasharray': '3 3' }, frame.g);
  }
}

function drawPhaseStrip(svg, r) {
  const ph = state.design.response.phaseUnwrapped;
  let lo = Infinity, hi = -Infinity;
  for (const v of ph) { if (Number.isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); } }
  const pad = (hi - lo) * 0.05 || 1;
  const frame = renderStrip(svg, r, [0, 1], [lo - pad, hi + pad], {
    title: '展开相位 rad（线性相位应为直线）',
    fmtX: () => '', fmtY: (v) => Math.round(v), yTicks: 4,
  });
  const dd = pathFromSeries(ph, frame.sx, frame.sy, 0, 1);
  svgEl('path', { d: dd, fill: 'none', stroke: '#3ddc97', 'stroke-width': 1.8 }, frame.g);
}

function drawGdStrip(svg, r) {
  const nominal = state.design.groupDelaySamples;
  const span = 2;
  const lo = nominal - span, hi = nominal + span;
  const frame = renderStrip(svg, r, [0, 1], [lo, hi], {
    title: `群延迟 样本（标称 ${nominal}，超出 ±${span} 的失真假点被裁到边界）`,
    fmtX: () => '', fmtY: (v) => v.toFixed(0), yTicks: 4,
  });
  svgEl('line', { x1: frame.x0, x2: frame.x1, y1: frame.sy(nominal), y2: frame.sy(nominal), stroke: '#ffd166', 'stroke-dasharray': '5 4' }, frame.g);
  const gd = state.design.response.gd.map((v) => (v == null ? null : clamp(v, lo, hi)));
  const dd = pathFromSeries(gd, frame.sx, frame.sy, 0, 1);
  svgEl('path', { d: dd, fill: 'none', stroke: '#c792ea', 'stroke-width': 1.8 }, frame.g);
}

/* ---------------- 取舍扫描 ---------------- */

async function runScan() {
  const btn = $('btn-scan');
  btn.disabled = true; btn.textContent = '扫描中…';
  await new Promise((r) => setTimeout(r, 30));
  try {
    const rows = scanLengthTradeoff(state.points, { sampleRate: state.sampleRate });
    const tb = $('tradeoff-body');
    clear(tb);
    for (const r of rows) {
      const tr = document.createElement('tr');
      tr.innerHTML = `<td>${r.N}</td><td>${r.groupDelaySamples}</td><td>${r.latencyMs.toFixed(2)}</td>` +
        `<td>${r.transitionHz ? Math.round(r.transitionHz) : '—'}</td>` +
        `<td>${r.maxErrorDb != null ? r.maxErrorDb.toFixed(2) : '—'}</td>` +
        `<td>${r.stopFloorDb != null ? r.stopFloorDb.toFixed(1) : '—'}</td>`;
      tb.appendChild(tr);
    }
    drawBars($('tradeoff-chart'), rows);
  } finally {
    btn.disabled = false; btn.textContent = '扫描 N=33…1025';
  }
}

/* ---------------- 自检 ---------------- */

function runSelfTestUi() {
  const r = runVerification();
  $('self-test-summary').innerHTML = r.pass
    ? `<span style="color:var(--good)">全部通过 ${r.passed}/${r.total}</span>`
    : `<span style="color:var(--bad)">${r.passed}/${r.total} 通过，存在失败项</span>`;
  const ul = $('self-test-list');
  clear(ul);
  for (const x of r.results) {
    const li = document.createElement('li');
    li.className = x.pass ? 'pass' : 'fail';
    li.innerHTML = `${x.pass ? '✓' : '✗'} ${x.title}<span class="d">${x.detail || ''}</span>`;
    ul.appendChild(li);
  }
  window.__lastVerification = r;
  return r;
}

/* ---------------- 双路径会话 ---------------- */

function modeLabel(m) {
  return { deferred: '延后交接', 'dual-old': '双核·旧尾声', 'dual-new': '双核·新时标' }[m] || m;
}

async function runSessionUi() {
  const btn = $('btn-run-session');
  btn.disabled = true;
  $('echo-status').textContent = '等候中…';
  try {
    const cfg = collectSessionConfig();
    const result = await runSession(cfg);
    if (!result.ok) {
      $('echo-status').textContent = '失败：' + result.error;
      return;
    }
    state.session = result;
    window.__lastSession = result;

    const sr = result.config.sampleRate;
    for (const key of ['echo', 'iem']) {
      if (state.urls[key]) URL.revokeObjectURL(state.urls[key]);
      const bytes = encodeWavBytes([result._audio[key]], sr);
      const blob = new Blob([bytes], { type: 'audio/wav' });
      state.urls[key] = URL.createObjectURL(blob);
      $('audio-' + key).src = state.urls[key];
      $('dl-' + key).href = state.urls[key];
    }

    const e = result.echo, im = result.iem, d = result.difference;
    $('echo-status').textContent = e.status === 'returned' ? `已回印（${modeLabel(e.mode)}）` : '完成';
    $('echo-await').textContent = e.awaitMs != null ? e.awaitMs + ' ms' : '同步';
    $('echo-peak').textContent = String(e.peakSampleFile);
    $('echo-beat').textContent = `时钟 ${e.peakSampleClock} / ${e.peakBeat.toFixed(4)} 拍`;
    $('echo-length').textContent = `${e.durationSamples} 样本 / ${e.durationSec.toFixed(2)}s`;

    $('iem-blocks').textContent = `${im.blocks} 块 × ${im.blockSize} = ${im.clockSamples} 样本`;
    $('iem-switch').textContent = `${im.switchLatchedAt}（量化 +${im.switchQuantizationSamples}）`;
    $('iem-peak').textContent = String(im.peakSampleFile);
    $('iem-beat').textContent = `时钟 ${im.peakSampleClock} / ${im.peakBeat.toFixed(4)} 拍`;
    $('iem-length').textContent = `${im.durationSamples} 样本 / ${im.durationSec.toFixed(2)}s`;

    $('beat-diff').textContent = `${d.absoluteBeats.toFixed(5)} 拍`;
    $('sample-diff').textContent = `${d.absoluteSamples}（带符号 ${d.signedSamples}）`;
    $('ms-diff').textContent = d.absoluteMs.toFixed(3) + ' ms';
    $('delta-gd').textContent = `${result.kernel.deltaGd} 样本（${result.kernel.gd1}→${result.kernel.gd2}）`;

    const setComb = (el, c, aligned) => {
      if (c.detected) {
        el.className = 'bad';
        el.textContent = `检出：${c.notchCount} 个陷波，最深 ${c.deepestDepthDb?.toFixed?.(1) ?? c.deepestDepthDb} dB，间隔 ${c.medianSpacingHz ? c.medianSpacingHz.toFixed(1) : '?'}Hz`;
      } else {
        el.className = 'good';
        el.textContent = aligned ? '无周期性深陷波（对齐有效）' : `未达判定阈值（陷波 ${c.notchCount}）`;
      }
    };
    setComb($('comb-raw'), result.comb.raw, false);
    setComb($('comb-aligned'), result.comb.aligned, true);

    $('diff-explain').textContent = explainDiff(cfg, d.absoluteSamples, result.kernel.deltaGd);

    requestAnimationFrame(() => {
      drawWaveform($('wave-echo'), result._audio.echo, {
        color: '#4fb4ff', sampleRate: sr,
        markers: [
          { sample: e.peakSampleFile, color: '#ffd166', label: `peak ${e.peakSampleFile}` },
          { sample: e.prerollSamples + e.switchAt, color: '#3ddc97', dash: [3, 3], label: 'S' },
        ],
      });
      drawWaveform($('wave-iem'), result._audio.iem, {
        color: '#c792ea', sampleRate: sr,
        markers: [
          { sample: im.peakSampleFile, color: '#ffd166', label: `peak ${im.peakSampleFile}` },
          { sample: im.prerollSamples + im.switchAt, color: '#3ddc97', dash: [3, 3], label: 'S_iem' },
        ],
      });
    });
    window.__lastSessionInfo = stripAudio(result);
  } finally {
    btn.disabled = false;
  }
}

function stripAudio(r) {
  const { _audio, ...info } = r;
  return info;
}

function explainDiff(cfg, diff, delta) {
  if (diff === 0) {
    if (cfg.echoMode === cfg.iemMode) return `两条路径使用相同的“${modeLabel(cfg.echoMode)}”对齐方式，实测峰位一致，绝对拍差为 0。`;
    return '该情景下两种对齐约定恰好把目标峰放在同一时间网格，拍差为 0；可切换到另一情景观察 Δgd 平移。';
  }
  if (diff === delta) {
    return `实测峰差恰好等于核长变化带来的群延迟差 Δgd=${delta} 样本：一条路径保留旧时标/旧尾声，另一条切换到新时标（或反向对齐）。这是从两条真实 WAV 量出的，移动显示标尺无法改变它。`;
  }
  return `实测峰差 ${diff} 样本，不等于 Δgd=${delta}，可能受到块时钟量化或交叉淡化区影响。`;
}

/* ---------------- 实时试听 AudioWorklet ---------------- */

let actx = null, node = null, wetGain = null, dryGain = null, srcNode = null;

async function ensureAudio() {
  if (!actx) {
    const AC = window.AudioContext || window.webkitAudioContext;
    actx = new AC();
    await actx.audioWorklet.addModule('/worklet/convolver-worklet.js');
    node = new AudioWorkletNode(actx, 'linear-phase-convolver', { outputChannelCount: [2] });
    wetGain = actx.createGain();
    dryGain = actx.createGain();
    node.connect(wetGain).connect(actx.destination);
    dryGain.connect(actx.destination);
    node.port.onmessage = (e) => {
      if (e.data.type === 'kernel-applied') {
        $('audio-status').textContent = `Worklet 卷积中 N=${e.data.length} @ ${e.data.sampleRate}Hz`;
        $('audio-status').className = 'badge good';
      }
    };
  }
  if (actx.state === 'suspended') await actx.resume();
  postWorkletKernel();
  return actx;
}

function postWorkletKernel() {
  if (!actx || !node || !state.design) return;
  // 用 AudioContext 自己的采样率重新设计（归一化频率目标不变）
  const d = designFir({
    points: state.points, N: state.N, firType: state.firType,
    window: state.window, stopbandDb: Number($('input-stopband').value),
    sampleRate: actx.sampleRate,
  });
  if (d.ok) node.port.postMessage({ type: 'kernel', kernel: Float32Array.from(d.kernel) });
}

async function playAudition() {
  try {
    const ctx = await ensureAudio();
    stopSource();
    let buffer;
    if (state.uploadedBuffer) {
      buffer = state.uploadedBuffer;
    } else {
      const x = generateMaterial(state.auditionMaterial, ctx.sampleRate, { bpm: Number($('input-bpm').value) || 124 });
      buffer = ctx.createBuffer(1, x.length, ctx.sampleRate);
      buffer.getChannelData(0).set(x);
    }
    srcNode = ctx.createBufferSource();
    srcNode.buffer = buffer;
    srcNode.connect(node);
    srcNode.connect(dryGain);
    applyWetMix();
    srcNode.onended = () => { /* 自然结束保留状态 */ };
    srcNode.start();
    $('audio-status').textContent = state.design ? `实时卷积 N=${state.design.N} 播放中…` : '无有效内核（当前为干声）';
    $('audio-status').className = 'badge ' + (state.design ? 'good' : 'bad');
  } catch (err) {
    $('audio-status').textContent = 'AudioWorklet 启动失败：' + err.message;
    $('audio-status').className = 'badge bad';
  }
}

function applyWetMix() {
  if (!actx) return;
  const wet = $('chk-wet').checked;
  wetGain.gain.setTargetAtTime(wet ? 1 : 0, actx.currentTime, 0.01);
  dryGain.gain.setTargetAtTime(wet ? 0 : 1, actx.currentTime, 0.01);
}

function stopSource() {
  if (srcNode) {
    try { srcNode.onended = null; srcNode.stop(); } catch { /* 已停止 */ }
    try { srcNode.disconnect(); } catch { /* noop */ }
    srcNode = null;
  }
}

/* ---------------- WAV 导出 ---------------- */

function downloadBytes(bytes, name, mime = 'audio/wav') {
  const blob = new Blob([bytes], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 4000);
}

async function exportWav() {
  $('export-status').textContent = '离线卷积中…';
  await new Promise((r) => setTimeout(r, 30));
  try {
    let channels, sr, tag;
    if (state.uploadedBuffer) {
      sr = state.uploadedBuffer.sampleRate;
      const d = designFromParams({ ...collectParams(), sampleRate: sr });
      if (!d.ok) throw new Error(d.message);
      const { streamConvolve } = await import('/dsp/convolution.js');
      channels = [];
      for (let c = 0; c < state.uploadedBuffer.numberOfChannels; c++) {
        channels.push(streamConvolve(state.uploadedBuffer.getChannelData(c), Float64Array.from(d.kernel), 128));
      }
      tag = 'upload';
    } else {
      sr = state.sampleRate;
      const x = generateMaterial(state.auditionMaterial, sr, { bpm: Number($('input-bpm').value) || 124 });
      const r = processMaterial({ ...collectParams(), material: state.auditionMaterial }, x);
      if (!r.ok) throw new Error(r.design.message);
      channels = [r.output];
      tag = state.auditionMaterial;
    }
    const bytes = encodeWavBytes(channels, sr);
    downloadBytes(bytes, `lpeq-${tag}-N${state.N}-${sr}Hz.wav`);
    $('export-status').textContent = `已导出 ${channels[0].length} 样本（x+N-1，完整尾部）`;
  } catch (err) {
    $('export-status').textContent = '导出失败：' + err.message;
  }
}

/* ---------------- 参数文件 ---------------- */

function saveParamFile() {
  const payload = { app: 'linear-phase-eq', version: 1, params: collectParams(), session: collectSessionConfig() };
  downloadBytes(new TextEncoder().encode(JSON.stringify(payload, null, 2)), 'lpeq-params.json', 'application/json');
}

function loadParamFile(file) {
  const reader = new FileReader();
  reader.onload = () => {
    try {
      const json = JSON.parse(reader.result);
      applyParams(json.params || json);
      if (json.session) {
        for (const [id, key] of [
          ['select-scenario', 'scenario'], ['input-N1', 'N1'], ['input-N2', 'N2'],
          ['input-bpm', 'bpm'], ['input-iemBlock', 'iemBlock'], ['select-session-material', 'material'],
          ['select-echo-mode', 'echoMode'], ['select-iem-mode', 'iemMode'],
        ]) if (json.session[key] != null) $(id).value = json.session[key];
        saveLocal();
      }
    } catch (e) {
      alert('参数文件解析失败：' + e.message);
    }
  };
  reader.readAsText(file);
}

/* ---------------- 杂项 UI ---------------- */

function updateKaiserVisibility() {
  $('kaiser-label').style.opacity = state.window === 'kaiser' ? '1' : '0.45';
}

async function checkHealth() {
  try {
    const r = await fetch('/api/health');
    const j = await r.json();
    $('health-dot').className = 'dot ok';
    $('health-text').textContent = j.ok ? '服务正常 · /api 可用' : '服务异常';
  } catch {
    $('health-dot').className = 'dot bad';
    $('health-text').textContent = '静态模式（API 不可用，核心仍可运行）';
  }
}

function bindEvents() {
  $('input-N').addEventListener('input', () => { $('val-N').textContent = $('input-N').value; redesign(); });
  $('select-firType').addEventListener('change', () => {
    // 切类型时把长度调整为对应奇偶
    const t = $('select-firType').value;
    const n = Number($('input-N').value);
    const oddType = t === 'I' || t === 'III';
    if (oddType && n % 2 === 0) $('input-N').value = n + 1;
    if (!oddType && n % 2 === 1) $('input-N').value = n + 1;
    redesign();
  });
  $('select-window').addEventListener('change', () => { updateKaiserVisibility(); redesign(); });
  $('input-stopband').addEventListener('input', () => { $('val-stopband').textContent = $('input-stopband').value + ' dB'; redesign(); });
  $('select-sampleRate').addEventListener('change', redesign);
  ['spec-transition', 'spec-error'].forEach((id) => $(id).addEventListener('input', evaluateSpec));
  ['resp-mag', 'resp-phase', 'resp-gd'].forEach((id) => $(id).addEventListener('change', renderResponse));

  $('btn-compare').addEventListener('click', () => {
    if (state.design) {
      state.compare = state.design;
      renderResponse();
    }
  });
  $('btn-compare-clear').addEventListener('click', () => { state.compare = null; renderResponse(); });

  $('btn-scan').addEventListener('click', runScan);
  $('btn-self-test').addEventListener('click', runSelfTestUi);
  $('btn-run-session').addEventListener('click', runSessionUi);

  $('material-row').addEventListener('click', (e) => {
    const b = e.target.closest('[data-material]');
    if (b) state.auditionMaterial = b.dataset.material;
  });
  $('file-audio').addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    try {
      const buf = await (actx || new AudioContext()).decodeAudioData(await file.arrayBuffer());
      state.uploadedBuffer = buf;
      $('audio-status').textContent = `已载入 ${file.name}（${buf.numberOfChannels} 声道, ${buf.sampleRate}Hz）`;
    } catch (err) {
      $('audio-status').textContent = '音频解码失败：' + err.message;
    }
  });
  $('btn-play').addEventListener('click', playAudition);
  $('btn-stop').addEventListener('click', stopSource);
  $('chk-wet').addEventListener('change', applyWetMix);
  $('btn-export-wav').addEventListener('click', exportWav);

  $('btn-save-params').addEventListener('click', saveParamFile);
  $('btn-load-params').addEventListener('click', () => $('file-params').click());
  $('file-params').addEventListener('change', (e) => e.target.files[0] && loadParamFile(e.target.files[0]));
  $('btn-reset-params').addEventListener('click', () => {
    localStorage.removeItem(STORE_KEY);
    state.points = defaultTargetPoints();
    $('input-N').value = 255;
    $('select-firType').value = 'I';
    $('select-window').value = 'hann';
    $('input-stopband').value = 70;
    $('val-stopband').textContent = '70 dB';
    $('select-sampleRate').value = '48000';
    redesign();
  });

  let resizeT = 0;
  window.addEventListener('resize', () => {
    clearTimeout(resizeT);
    resizeT = setTimeout(() => {
      renderEq();
      if (state.session) {
        const r = state.session;
        const sr = r.config.sampleRate;
        requestAnimationFrame(() => {
          const e = r.echo, im = r.iem;
          drawWaveform($('wave-echo'), r._audio.echo, {
            color: '#4fb4ff', sampleRate: sr,
            markers: [
              { sample: e.peakSampleFile, color: '#ffd166', label: `peak ${e.peakSampleFile}` },
              { sample: e.prerollSamples + e.switchAt, color: '#3ddc97', dash: [3, 3], label: 'S' },
            ],
          });
          drawWaveform($('wave-iem'), r._audio.iem, {
            color: '#c792ea', sampleRate: sr,
            markers: [
              { sample: im.peakSampleFile, color: '#ffd166', label: `peak ${im.peakSampleFile}` },
              { sample: im.prerollSamples + im.switchAt, color: '#3ddc97', dash: [3, 3], label: 'S_iem' },
            ],
          });
        });
      }
    }, 250);
  });
}

/* ---------------- 启动 ---------------- */

function exposeApi() {
  window.LPEQ = {
    version: '1.0.0',
    designFir,
    scanLengthTradeoff,
    runVerification,
    runSession,
    get state() { return state; },
    redesign,
    async runSessionNow() { return runSessionUi(); },
    async selfTest() { return runSelfTestUi(); },
  };
  window.__lpeqReady = true;
}

function init() {
  applyPersisted();
  bindEvents();
  initEqInteractions();
  checkHealth();
  updateKaiserVisibility();
  redesign();
  exposeApi();
  // 首屏即给出真实会话测量与自检，避免空面板
  setTimeout(() => {
    runSessionUi();
    runSelfTestUi();
  }, 60);
}

if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
else init();
