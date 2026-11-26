// 界面接线冒烟：用 jsdom 加载真实 index.html 与真实 js/app.js（仅把 /dsp 绝对
// 导入改写为相对路径的临时入口），验证首屏渲染、自动会话、自检与控件反馈。
// 未安装 jsdom 时跳过：核心 DSP/Worklet 测试不依赖它。
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync, rmSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.join(__dirname, '..');

let JSDOM = null;
try {
  ({ JSDOM } = await import('jsdom'));
} catch {
  // jsdom 未安装
}

test('Web 面板接线冒烟（jsdom）', { skip: !JSDOM, timeout: 30000 }, async (t) => {
  const html = readFileSync(path.join(ROOT, 'public', 'index.html'), 'utf8');
  const dom = new JSDOM(html, { url: 'http://localhost:3000/', pretendToBeVisual: true });
  const { window } = dom;

  const setGlobal = (k, v) => {
    try { globalThis[k] = v; }
    catch { Object.defineProperty(globalThis, k, { configurable: true, value: v }); }
  };
  setGlobal('window', window);
  setGlobal('document', window.document);
  setGlobal('navigator', window.navigator);
  setGlobal('HTMLElement', window.HTMLElement);
  setGlobal('HTMLCanvasElement', window.HTMLCanvasElement);
  setGlobal('SVGElement', window.SVGElement);
  setGlobal('Event', window.Event);
  setGlobal('MouseEvent', window.MouseEvent);
  setGlobal('Blob', window.Blob || globalThis.Blob);
  setGlobal('localStorage', window.localStorage);

  const raf = (cb) => setTimeout(() => cb(performance.now()), 0);
  setGlobal('requestAnimationFrame', raf);
  setGlobal('cancelAnimationFrame', (id) => clearTimeout(id));
  window.requestAnimationFrame = raf;

  const noopCtx = new Proxy({}, {
    get(_t, p) {
      if (p === 'measureText') return () => ({ width: 10 });
      if (p === 'createLinearGradient') return () => ({ addColorStop() {} });
      return typeof p === 'string' ? () => {} : undefined;
    },
    set() { return true; },
  });
  window.HTMLCanvasElement.prototype.getContext = () => noopCtx;
  Object.defineProperty(window.HTMLCanvasElement.prototype, 'clientWidth', { configurable: true, value: 900 });
  Object.defineProperty(window.HTMLCanvasElement.prototype, 'clientHeight', { configurable: true, value: 150 });

  window.URL.createObjectURL = () => 'blob:mock';
  window.URL.revokeObjectURL = () => {};
  try {
    globalThis.URL.createObjectURL = () => 'blob:mock';
    globalThis.URL.revokeObjectURL = () => {};
  } catch {
    Object.defineProperty(globalThis.URL, 'createObjectURL', { configurable: true, writable: true, value: () => 'blob:mock' });
  }

  let src = readFileSync(path.join(ROOT, 'public', 'js', 'app.js'), 'utf8');
  src = src
    .replaceAll("from '/dsp/", "from '../dsp/")
    .replaceAll("import('/dsp/", "import('../dsp/");
  const tmp = path.join(ROOT, 'public', 'js', `__smoke_app_${process.pid}.mjs`);
  writeFileSync(tmp, src);

  const unhandled = [];
  const onRej = (e) => unhandled.push(e);
  process.on('unhandledRejection', onRej);

  const txt = (id) => window.document.getElementById(id)?.textContent || '';
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const setSel = (id, v) => {
    const el = window.document.getElementById(id);
    el.value = v;
    el.dispatchEvent(new window.Event('change', { bubbles: true }));
  };

  try {
    await import(pathToFileURL(tmp).href + '?t=' + Date.now());
    await sleep(1500);

    assert.equal(window.__lpeqReady, true, '首屏应置 __lpeqReady');
    assert.match(txt('metric-gd'), /\d+/, '群延迟指标应渲染');
    assert.ok(window.document.querySelectorAll('#eq-svg circle').length >= 2, 'EQ 拖点应渲染');
    assert.ok(window.document.querySelectorAll('#eq-svg path').length >= 2, 'EQ 曲线应渲染');
    assert.ok(window.document.querySelectorAll('#response-svg path').length >= 3, '响应三条带应渲染');
    const nanPaths = [...window.document.querySelectorAll('#eq-svg path, #response-svg path')]
      .filter((p) => /NaN|Infinity/.test(p.getAttribute('d') || ''));
    assert.equal(nanPaths.length, 0, '所有 SVG path 坐标必须有限（目标点 {f,db} 映射回归）');
    assert.match(txt('beat-diff'), /0\.019/, '自动会话应给出 ~0.019 拍差');
    assert.ok(txt('sample-diff').startsWith('448'), '样本差应为 448');
    assert.match(txt('self-test-summary'), /13\/13/, '浏览器内自检应 13/13');
    assert.match(txt('comb-raw'), /检出/, '未对齐应检出梳状');
    assert.match(txt('comb-aligned'), /无周期|对齐有效/, '对齐后不应误报');
    assert.ok((window.document.getElementById('audio-echo').getAttribute('src') || '').startsWith('blob:'), '回印 audio 应有 WAV blob');

    setSel('select-scenario', 'new-effective');
    window.document.getElementById('btn-run-session').click();
    await sleep(500);
    assert.ok(txt('sample-diff').startsWith('0（'), '情景 B 默认策略峰差应为 0');

    window.LPEQ.state.points = [{ f: 0, db: 0 }, { f: 1, db: 0 }];
    setSel('select-firType', 'II');
    await sleep(150);
    assert.match(window.document.getElementById('kernel-feedback').className, /show error/, 'II 型+平直应显示失败反馈');
    assert.ok(window.document.getElementById('btn-fix-type'), '应给出自动修复按钮');
    window.document.getElementById('btn-fix-type').click();
    await sleep(150);
    assert.equal(window.document.getElementById('select-firType').value, 'I', '修复应切到 I 型');

    const n = window.document.getElementById('input-N');
    n.value = '511';
    n.dispatchEvent(new window.Event('input', { bubbles: true }));
    await sleep(150);
    assert.equal(txt('metric-gd'), '255', 'N=511 群延迟应为 255');

    window.document.getElementById('btn-scan').click();
    await sleep(2000);
    assert.ok(window.document.querySelectorAll('#tradeoff-body tr').length >= 5, '权衡表至少 5 行');
    assert.ok(window.document.querySelectorAll('#tradeoff-chart rect').length >= 5, '权衡柱状图应渲染 rect');

    await sleep(350);
    const stored = JSON.parse(window.localStorage.getItem('lpeq-params-v1') || 'null');
    assert.ok(stored?.params && stored.params.N === 511, '参数应自动保存到 localStorage');

    assert.deepEqual(unhandled, [], '不应有未处理 Promise 拒绝');
  } finally {
    process.removeListener('unhandledRejection', onRej);
    rmSync(tmp, { force: true });
    window.close();
  }
});
