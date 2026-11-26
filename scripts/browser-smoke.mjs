// 真实 Chromium 端到端冒烟（可选，不被 `npm test` 默认执行）。
//
// 准备：
//   npm i -D playwright
//   npx playwright install chromium
//   npm start                # 另一个终端，默认 :3000
//   BASE=http://localhost:3000 npm run smoke:browser
//
// 无 root 的 Debian/arm64 容器若缺 .so，可 apt-get download 对应 deb 并解包，
// 再用 LD_LIBRARY_PATH=<解包前缀>/lib/aarch64-linux-gnu:<...>/usr/lib/aarch64-linux-gnu 运行。
// 脚本会验证首屏渲染、自动双路径会话、自检、失败反馈、扫描、真实拖拽/双击、
// AudioWorklet 实时卷积状态与 WAV 下载，并要求浏览器控制台 0 error。
import { mkdirSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

let chromium;
try {
  ({ chromium } = await import('playwright'));
} catch {
  console.error('未安装 playwright。请先运行：npm i -D playwright && npx playwright install chromium');
  process.exit(2);
}

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const SHOT_DIR = path.join(__dirname, '..', 'output', 'browser-smoke');
mkdirSync(SHOT_DIR, { recursive: true });

const BASE = process.env.BASE || 'http://localhost:3000';
const errors = [];
const browser = await chromium.launch({
  headless: true,
  args: ['--no-sandbox', '--autoplay-policy=no-user-gesture-required', '--use-fake-ui-for-media-stream', '--use-fake-audio-stream'],
});
const page = await browser.newPage({ viewport: { width: 1600, height: 1050 } });
page.on('console', (m) => { if (m.type() === 'error') errors.push('console.error: ' + m.text()); });
page.on('pageerror', (e) => errors.push('pageerror: ' + e.message));

const results = [];
const check = (name, cond, extra = '') => results.push({ name, pass: !!cond, extra: String(extra) });

await page.goto(BASE, { waitUntil: 'networkidle' });
await page.waitForTimeout(800);

check('window.__lpeqReady', await page.evaluate(() => window.__lpeqReady === true));
check('LPEQ API exposed', await page.evaluate(() => !!window.LPEQ));

const gd = await page.textContent('#metric-gd');
check('group delay metric rendered', /\d/.test(gd), gd);
check('symmetry residual rendered', (await page.textContent('#metric-sym')).length > 0);

check('EQ handles drawn', await page.locator('#eq-svg circle').count() >= 2);
check('EQ curves drawn', await page.locator('#eq-svg path').count() >= 2);
check('response strips drawn', await page.locator('#response-svg path').count() >= 2);

await page.waitForFunction(() => /\d\.\d+/.test(document.querySelector('#beat-diff')?.textContent || ''), null, { timeout: 8000 });
const beat = await page.textContent('#beat-diff');
const sampleDiff = await page.textContent('#sample-diff');
check('auto session beat diff ~0.019', /0\.019/.test(beat), beat);
check('sample diff = 448', sampleDiff.includes('448'), sampleDiff);
const echoPeak = Number(await page.textContent('#echo-peak'));
const iemPeak = Number(await page.textContent('#iem-peak'));
check('peaks differ by 448', iemPeak - echoPeak === 448, `${echoPeak} vs ${iemPeak}`);

await page.waitForFunction(() => /全部通过 13\/13/.test(document.querySelector('#self-test-summary')?.textContent || ''), null, { timeout: 8000 });
check('browser self-test 13/13', true, (await page.textContent('#self-test-summary')).trim());

await page.selectOption('#select-scenario', 'new-effective');
await page.click('#btn-run-session');
await page.waitForFunction(() => document.querySelector('#sample-diff')?.textContent?.startsWith('0（'), null, { timeout: 8000 });
check('scenario B default diff 0', true, await page.textContent('#sample-diff'));

await page.selectOption('#select-iem-mode', 'dual-old');
await page.click('#btn-run-session');
await page.waitForFunction(() => document.querySelector('#sample-diff')?.textContent?.startsWith('448'), null, { timeout: 8000 });
check('B + iem dual-old diff 448', true, await page.textContent('#sample-diff'));

await page.click('[data-preset="flat"]');
await page.selectOption('#select-firType', 'II');
await page.waitForTimeout(300);
const fbClass = await page.getAttribute('#kernel-feedback', 'class');
const fbText = await page.textContent('#kernel-feedback');
check('type2 Nyquist failure feedback', /show error/.test(fbClass) && /Nyquist/.test(fbText), fbText.slice(0, 90));
check('fix suggestion button', await page.locator('#btn-fix-type').count() === 1);
await page.click('#btn-fix-type');
await page.waitForTimeout(200);
check('fix switches to type I', await page.inputValue('#select-firType') === 'I');

await page.evaluate(() => {
  const el = document.querySelector('#input-N');
  Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set.call(el, '511');
  el.dispatchEvent(new Event('input', { bubbles: true }));
});
await page.waitForTimeout(300);
check('N slider redesign', await page.textContent('#val-N') === '511');
check('gd = 255 for N511', await page.textContent('#metric-gd') === '255');

await page.click('#btn-scan');
await page.waitForFunction(() => document.querySelectorAll('#tradeoff-body tr').length >= 5, null, { timeout: 15000 });
check('tradeoff rows >=5', await page.locator('#tradeoff-body tr').count() >= 5);
await page.waitForFunction(() => document.querySelectorAll('#tradeoff-chart rect').length >= 5, null, { timeout: 5000 });
check('tradeoff bar rects >=5', await page.locator('#tradeoff-chart rect').count() >= 5);

const nanPaths = await page.evaluate(() => {
  const bad = [];
  document.querySelectorAll('#eq-svg path, #response-svg path').forEach((p, i) => {
    const d = p.getAttribute('d') || '';
    if (d.includes('NaN') || d.includes('Infinity')) bad.push(i + ':' + d.slice(0, 40));
  });
  return bad;
});
check('no NaN/Infinity svg paths', nanPaths.length === 0, nanPaths.slice(0, 3).join(' | '));

await page.locator('#eq-svg').scrollIntoViewIfNeeded();
await page.waitForTimeout(150);
const box = await page.locator('#eq-svg circle').first().boundingBox();
const dbBeforeDrag = await page.evaluate(() => window.LPEQ.state.points[0].db);
await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
await page.mouse.down();
await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2 + 60, { steps: 6 });
await page.mouse.up();
await page.waitForTimeout(300);
const dbAfterDrag = await page.evaluate(() => window.LPEQ.state.points[0].db);
check('drag changes point gain', dbAfterDrag < dbBeforeDrag - 5, `${dbBeforeDrag} -> ${dbAfterDrag}`);
check('drag keeps valid kernel', !/error/.test(await page.getAttribute('#kernel-feedback', 'class') || ''));

await page.click('#btn-play').catch(() => {});
await page.waitForTimeout(1500);
const audioStatus = await page.textContent('#audio-status');
check('audioworklet live convolution', /Worklet 卷积中|播放中/.test(audioStatus), audioStatus);

const wavInfo = await page.evaluate(async () => {
  const m = await import('/dsp/wav.js');
  const b = m.encodeWavBytes([new Float64Array(100)], 48000);
  return { len: b.length, four: String.fromCharCode(b[0], b[1], b[2], b[3]) };
});
check('browser wav encoder', wavInfo.len === 244 && wavInfo.four === 'RIFF', JSON.stringify(wavInfo));

const [download] = await Promise.all([
  page.waitForEvent('download', { timeout: 20000 }),
  page.click('#btn-export-wav'),
]);
const dlPath = path.join(SHOT_DIR, 'export.wav');
await download.saveAs(dlPath);
const wavBuf = readFileSync(dlPath);
check('export wav download RIFF', wavBuf.length > 1000 && wavBuf.subarray(0, 4).toString() === 'RIFF', `${wavBuf.length} bytes`);
check('export status reports tail length', /已导出/.test(await page.textContent('#export-status')), await page.textContent('#export-status'));

await page.check('#resp-phase');
await page.waitForTimeout(200);
const phase = await page.evaluate(() => {
  let bad = 0;
  document.querySelectorAll('#response-svg path').forEach((p) => {
    if ((p.getAttribute('d') || '').includes('NaN')) bad++;
  });
  return { bad, paths: document.querySelectorAll('#response-svg path').length };
});
check('phase strip finite paths', phase.bad === 0 && phase.paths >= 4, JSON.stringify(phase));

const circlesBefore = await page.locator('#eq-svg circle').count();
await page.locator('#eq-svg').scrollIntoViewIfNeeded();
await page.waitForTimeout(150);
const eqBox = await page.locator('#eq-svg').boundingBox();
const dblX = eqBox.x + eqBox.width * 0.55;
const dblY = eqBox.y + eqBox.height * 0.45;
await page.mouse.dblclick(dblX, dblY);
await page.waitForTimeout(250);
check('doubleclick adds EQ point', await page.locator('#eq-svg circle').count() === circlesBefore + 1,
  `${circlesBefore} -> ${await page.locator('#eq-svg circle').count()}`);

// 参数 JSON 保存
const [paramDl] = await Promise.all([
  page.waitForEvent('download', { timeout: 10000 }),
  page.click('#btn-save-params'),
]);
const paramPath = path.join(SHOT_DIR, 'params.json');
await paramDl.saveAs(paramPath);
const paramJson = JSON.parse(readFileSync(paramPath, 'utf8'));
check('save params JSON', paramJson.app === 'linear-phase-eq' && Array.isArray(paramJson.params.points), `N=${paramJson.params?.N}`);

// 参数 JSON 读取往返：载入 N=333 后群延迟必须变为 166
const customParamPath = path.join(SHOT_DIR, 'load-params.json');
const { writeFileSync } = await import('node:fs');
writeFileSync(customParamPath, JSON.stringify({
  app: 'linear-phase-eq', version: 1,
  params: { points: [{ f: 0, db: 2 }, { f: 0.5, db: -3 }, { f: 1, db: 1 }], N: 333, firType: 'I', window: 'hann', sampleRate: 48000 },
}));
await page.setInputFiles('#file-params', customParamPath);
await page.waitForFunction(() => document.querySelector('#metric-gd')?.textContent === '166', null, { timeout: 5000 });
check('load params JSON redesigns kernel', await page.textContent('#val-N') === '333' && await page.textContent('#metric-gd') === '166');

await page.screenshot({ path: path.join(SHOT_DIR, 'full.png'), fullPage: true }).catch(() => {});
await page.locator('#eq-svg').screenshot({ path: path.join(SHOT_DIR, 'eq.png') }).catch(() => {});
await page.locator('#response-svg').screenshot({ path: path.join(SHOT_DIR, 'response.png') }).catch(() => {});

console.log('\n==== BROWSER SMOKE ====');
let failed = 0;
for (const r of results) {
  console.log((r.pass ? 'PASS' : 'FAIL') + '  ' + r.name + (r.extra ? '  [' + r.extra + ']' : ''));
  if (!r.pass) failed++;
}
console.log(`\n${results.length - failed}/${results.length} passed; console/page errors: ${errors.length}`);
for (const e of errors.slice(0, 12)) console.log('  ' + e);
await browser.close();
process.exit(failed || errors.length ? 1 : 0);
