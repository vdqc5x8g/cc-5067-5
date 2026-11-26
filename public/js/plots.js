// 纯 DOM/SVG/Canvas 绘图辅助，无第三方依赖。

export const SVG_NS = 'http://www.w3.org/2000/svg';

export function svgEl(tag, attrs = {}, parent) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'text') node.textContent = v;
    else node.setAttribute(k, v);
  }
  if (parent) parent.appendChild(node);
  return node;
}

export function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

// 线性坐标框架。domain {x:[min,max], y:[min,max]}，返回比例尺与绘图区。
export function setupFrame(svg, W, H, margin, xDomain, yDomain, opts = {}) {
  clear(svg);
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  const x0 = margin.l, x1 = W - margin.r;
  const y0 = H - margin.b, y1 = margin.t;
  const sx = (x) => x0 + ((x - xDomain[0]) / (xDomain[1] - xDomain[0])) * (x1 - x0);
  const sy = (y) => y0 - ((y - yDomain[0]) / (yDomain[1] - yDomain[0])) * (y0 - y1);

  const grid = svgEl('g', {}, svg);
  const xTicks = opts.xTicks || 10;
  const yTicks = opts.yTicks || 8;
  const fmtX = opts.fmtX || ((v) => fmtTick(v));
  const fmtY = opts.fmtY || ((v) => fmtTick(v));

  for (let i = 0; i <= xTicks; i++) {
    const v = xDomain[0] + (i / xTicks) * (xDomain[1] - xDomain[0]);
    const x = sx(v);
    svgEl('line', { x1: x, x2: x, y1: y1, y2: y0, stroke: '#1c2534', 'stroke-width': 1 }, grid);
    const anchor = i === 0 ? 'start' : (i === xTicks ? 'end' : 'middle');
    const t = svgEl('text', { x, y: H - margin.b + 16, fill: '#8a97aa', 'font-size': 11, 'text-anchor': anchor }, grid);
    t.textContent = fmtX(v);
  }
  for (let i = 0; i <= yTicks; i++) {
    const v = yDomain[0] + (i / yTicks) * (yDomain[1] - yDomain[0]);
    const y = sy(v);
    svgEl('line', { x1: x0, x2: x1, y1: y, y2: y, stroke: '#1c2534', 'stroke-width': 1 }, grid);
    const t = svgEl('text', { x: x0 - 7, y: y + 4, fill: '#8a97aa', 'font-size': 11, 'text-anchor': 'end' }, grid);
    t.textContent = fmtY(v);
  }
  // 0 轴加重
  if (yDomain[0] < 0 && yDomain[1] > 0) {
    svgEl('line', { x1: x0, x2: x1, y1: sy(0), y2: sy(0), stroke: '#3a4a66', 'stroke-width': 1.2 }, grid);
  }
  svgEl('rect', { x: x0, y: y1, width: x1 - x0, height: y0 - y1, fill: 'none', stroke: '#2a3342' }, svg);
  return { sx, sy, x0, x1, y0, y1 };
}

// 由等间隔 x 采样（x 从 xMin 到 xMax，长度 ys.length）生成 path。
export function pathFromSeries(ys, sx, sy, xMin, xMax, yMapper) {
  const n = ys.length;
  let d = '';
  let started = false;
  for (let i = 0; i < n; i++) {
    const yv = yMapper ? yMapper(ys[i], i) : ys[i];
    if (yv == null || !Number.isFinite(yv)) { started = false; continue; }
    const xv = xMin + (i / (n - 1)) * (xMax - xMin);
    const x = sx(xv);
    const y = sy(yv);
    d += `${started ? 'L' : 'M'}${x.toFixed(2)},${y.toFixed(2)}`;
    started = true;
  }
  return d;
}

// 任意点序列 path；点支持 {x,y} 或 EQ 曲线的 {f,db}。
export function pathFromPoints(points, sx, sy) {
  return points
    .map((p, i) => {
      const px = p.x ?? p.f;
      const py = p.y ?? p.db;
      return `${i ? 'L' : 'M'}${sx(px).toFixed(2)},${sy(py).toFixed(2)}`;
    })
    .join('');
}

// 消除浮点刻度的长小数尾巴（如 -26.39999999999999 -> -26.4）。
export function fmtTick(v) {
  if (!Number.isFinite(v)) return '';
  const r = Math.round(v * 100) / 100;
  return String(Object.is(r, -0) ? 0 : r);
}

// 波形峰值绘制（min/max 抽稀），markers: [{sample,color,label}]
export function drawWaveform(canvas, audio, opts = {}) {
  const dpr = window.devicePixelRatio || 1;
  const cssW = canvas.clientWidth || 900;
  const cssH = canvas.clientHeight || 150;
  canvas.width = Math.round(cssW * dpr);
  canvas.height = Math.round(cssH * dpr);
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, cssW, cssH);
  ctx.fillStyle = '#0a0e15';
  ctx.fillRect(0, 0, cssW, cssH);

  const dur = audio.length;
  const xFor = (s) => (s / dur) * cssW;
  const mid = cssH / 2;
  const amp = cssH * 0.42;

  // 网格中线
  ctx.strokeStyle = '#1c2534';
  ctx.beginPath();
  ctx.moveTo(0, mid);
  ctx.lineTo(cssW, mid);
  ctx.stroke();

  // min/max 抽稀
  const buckets = Math.floor(cssW * 2);
  const per = Math.max(1, Math.floor(dur / buckets));
  ctx.strokeStyle = opts.color || '#4fb4ff';
  ctx.lineWidth = 1;
  ctx.beginPath();
  let x = 0;
  for (let b = 0; b < buckets; b++) {
    const i0 = b * per;
    if (i0 >= dur) break;
    let mn = 0, mx = 0;
    for (let i = i0; i < Math.min(dur, i0 + per); i++) {
      const v = audio[i];
      if (v < mn) mn = v;
      if (v > mx) mx = v;
    }
    x = (b / buckets) * cssW;
    ctx.moveTo(x, mid - mx * amp);
    ctx.lineTo(x, mid - mn * amp);
  }
  ctx.stroke();

  // 样本标尺对应的时间网格
  ctx.fillStyle = '#5f6d82';
  ctx.font = '10px ui-monospace, monospace';
  const sec = opts.sampleRate ? dur / opts.sampleRate : null;
  for (let i = 0; i <= 8; i++) {
    const xx = (i / 8) * cssW;
    ctx.strokeStyle = '#141b27';
    ctx.beginPath(); ctx.moveTo(xx, 0); ctx.lineTo(xx, cssH); ctx.stroke();
    if (sec) ctx.fillText(((i / 8) * sec).toFixed(2) + 's', xx + 3, cssH - 4);
  }

  // 标记
  for (const m of opts.markers || []) {
    const xx = xFor(m.sample);
    ctx.strokeStyle = m.color || '#ffd166';
    ctx.lineWidth = 1.6;
    ctx.setLineDash(m.dash || [5, 3]);
    ctx.beginPath(); ctx.moveTo(xx, 6); ctx.lineTo(xx, cssH - 18); ctx.stroke();
    ctx.setLineDash([]);
    if (m.label) {
      ctx.fillStyle = m.color || '#ffd166';
      ctx.font = '11px ui-monospace, monospace';
      ctx.fillText(m.label, xx + 5, 16);
    }
  }
}

// 在同一 SVG 的指定矩形区域内画一个坐标条带（不清空整个 SVG）。
export function renderStrip(svg, r, xDomain, yDomain, opts = {}) {
  const g = svgEl('g', {}, svg);
  const x0 = r.x + (r.ml ?? 52);
  const x1 = r.x + r.w - (r.mr ?? 14);
  const yTop = r.y + (r.mt ?? 10);
  const yBot = r.y + r.h - (r.mb ?? 22);
  const sx = (x) => x0 + ((x - xDomain[0]) / (xDomain[1] - xDomain[0])) * (x1 - x0);
  const sy = (y) => yBot - ((y - yDomain[0]) / (yDomain[1] - yDomain[0])) * (yBot - yTop);

  svgEl('rect', { x: r.x, y: r.y, width: r.w, height: r.h, fill: opts.bg || '#0b0f16', rx: 8 }, g);
  const xTicks = opts.xTicks ?? 10;
  const yTicks = opts.yTicks ?? 5;
  for (let i = 0; i <= xTicks; i++) {
    const v = xDomain[0] + (i / xTicks) * (xDomain[1] - xDomain[0]);
    const x = sx(v);
    svgEl('line', { x1: x, x2: x, y1: yTop, y2: yBot, stroke: '#1c2534' }, g);
    if (opts.xLabels !== false) {
      const anchor = i === 0 ? 'start' : (i === xTicks ? 'end' : 'middle');
      const t = svgEl('text', { x, y: r.y + r.h - 6, fill: '#8a97aa', 'font-size': 10, 'text-anchor': anchor }, g);
      t.textContent = opts.fmtX ? opts.fmtX(v) : v.toFixed(1);
    }
  }
  for (let i = 0; i <= yTicks; i++) {
    const v = yDomain[0] + (i / yTicks) * (yDomain[1] - yDomain[0]);
    const y = sy(v);
    svgEl('line', { x1: x0, x2: x1, y1: y, y2: y, stroke: '#1c2534' }, g);
    const t = svgEl('text', { x: x0 - 6, y: y + 3.5, fill: opts.yLabelColor || '#8a97aa', 'font-size': 10, 'text-anchor': 'end' }, g);
    t.textContent = opts.fmtY ? opts.fmtY(v) : String(Math.round(v));
  }
  if (yDomain[0] < 0 && yDomain[1] > 0) {
    svgEl('line', { x1: x0, x2: x1, y1: sy(0), y2: sy(0), stroke: '#3a4a66' }, g);
  }
  svgEl('text', { x: x0 + 6, y: yTop + 13, fill: opts.titleColor || '#dbe4f0', 'font-size': 11, 'font-weight': 700 }, g).textContent = opts.title || '';
  return { g, sx, sy, x0, x1, yTop, yBot };
}

// 简易柱状/折线迷你图（取舍扫描过渡带）
export function drawBars(svg, rows) {
  clear(svg);
  const W = 700, H = 120, m = { l: 44, r: 10, t: 12, b: 26 };
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  const vals = rows.map((r) => r.transitionHz || 0);
  const maxV = Math.max(...vals) * 1.1 || 1;
  const bw = (W - m.l - m.r) / rows.length;
  svgEl('line', { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, stroke: '#2a3342' }, svg);
  rows.forEach((r, i) => {
    const h = ((r.transitionHz || 0) / maxV) * (H - m.t - m.b);
    const x = m.l + i * bw + bw * 0.2;
    const y = H - m.b - h;
    svgEl('rect', { x, y, width: bw * 0.6, height: h, fill: '#4fb4ff', rx: 3 }, svg);
    const t = svgEl('text', { x: x + bw * 0.3, y: H - m.b + 15, fill: '#8a97aa', 'font-size': 10, 'text-anchor': 'middle' }, svg);
    t.textContent = r.N;
    const tv = svgEl('text', { x: x + bw * 0.3, y: y - 3, fill: '#dbe4f0', 'font-size': 9.5, 'text-anchor': 'middle' }, svg);
    tv.textContent = r.transitionHz ? Math.round(r.transitionHz) : '—';
  });
}
