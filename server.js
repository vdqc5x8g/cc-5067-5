// 零依赖启动：npm start  ->  http://localhost:3000
import http from 'node:http';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { designFromParams, processMaterial, scanLengthTradeoff } from './public/dsp/pipeline.js';
import { runSession, runSessionSync } from './public/dsp/session.js';
import { runVerification } from './public/dsp/verify.js';
import { encodeWavBuffer } from './public/dsp/wav.js';
import { defaultTargetPoints } from './public/dsp/target.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PUBLIC = path.join(__dirname, 'public');
const PORT = Number(process.env.PORT) || 3000;
const HOST = process.env.HOST || '0.0.0.0';

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.ico': 'image/x-icon',
  '.wav': 'audio/wav',
};

function sendJson(res, code, obj) {
  const body = JSON.stringify(obj);
  res.writeHead(code, { 'content-type': 'application/json; charset=utf-8' });
  res.end(body);
}

function readBody(req, limit = 8 * 1024 * 1024) {
  return new Promise((resolve, reject) => {
    let size = 0;
    const chunks = [];
    req.on('data', (c) => {
      size += c.length;
      if (size > limit) {
        reject(new Error('请求体过大'));
        req.destroy();
        return;
      }
      chunks.push(c);
    });
    req.on('end', () => {
      if (!chunks.length) return resolve({});
      try {
        resolve(JSON.parse(Buffer.concat(chunks).toString('utf8')));
      } catch (e) {
        reject(new Error('JSON 解析失败: ' + e.message));
      }
    });
    req.on('error', reject);
  });
}

async function serveStatic(req, res, url) {
  let pathname = decodeURIComponent(url.pathname);
  if (pathname === '/') pathname = '/index.html';
  // 禁止路径穿越
  const filePath = path.normalize(path.join(PUBLIC, pathname));
  if (!filePath.startsWith(PUBLIC + path.sep) && filePath !== path.join(PUBLIC, 'index.html')) {
    res.writeHead(403);
    return res.end('Forbidden');
  }
  try {
    const data = await readFile(filePath);
    const ext = path.extname(filePath).toLowerCase();
    res.writeHead(200, { 'content-type': MIME[ext] || 'application/octet-stream' });
    res.end(data);
  } catch {
    // 单页回退
    try {
      const index = await readFile(path.join(PUBLIC, 'index.html'));
      res.writeHead(200, { 'content-type': MIME['.html'] });
      res.end(index);
    } catch {
      res.writeHead(404);
      res.end('Not found');
    }
  }
}

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, `http://${req.headers.host || 'localhost'}`);
  const p = url.pathname;

  try {
    if (p === '/api/health') {
      return sendJson(res, 200, { ok: true, app: 'linear-phase-eq', time: new Date().toISOString() });
    }

    if (p === '/api/design' && req.method === 'POST') {
      const body = await readBody(req);
      const result = designFromParams(body);
      return sendJson(res, result.ok ? 200 : 422, result);
    }

    if (p === '/api/scan' && req.method === 'POST') {
      const body = await readBody(req);
      const rows = scanLengthTradeoff(body.points || defaultTargetPoints(), body.options || {});
      return sendJson(res, 200, { ok: true, rows });
    }

    if (p === '/api/verify') {
      return sendJson(res, 200, runVerification());
    }

    if (p === '/api/session' && req.method === 'POST') {
      const body = await readBody(req);
      const result = await runSession(body);
      if (!result.ok) return sendJson(res, 422, result);
      const { _audio, ...info } = result;
      void _audio;
      return sendJson(res, 200, info);
    }

    if (p === '/api/demo/measurements') {
      const cfg = {
        scenario: url.searchParams.get('scenario') || 'old-tail',
        echoMode: url.searchParams.get('echoMode') || undefined,
        iemMode: url.searchParams.get('iemMode') || undefined,
        material: url.searchParams.get('material') || 'click',
      };
      const s = runSessionSync(cfg);
      return sendJson(res, 200, {
        ok: true,
        config: cfg,
        echo: s.echoMeasure,
        iem: s.iemMeasure,
        difference: s.difference,
        comb: s.comb,
        kernel: { gd1: s.resolved.gd1, gd2: s.resolved.gd2, deltaGd: s.resolved.deltaGd },
      });
    }

    if (p === '/api/demo/session.wav') {
      const which = url.searchParams.get('path') === 'iem' ? 'iemAudio' : 'echoAudio';
      const cfg = {
        scenario: url.searchParams.get('scenario') || 'old-tail',
        echoMode: url.searchParams.get('echoMode') || undefined,
        iemMode: url.searchParams.get('iemMode') || undefined,
        material: url.searchParams.get('material') || 'click',
      };
      const s = runSessionSync(cfg);
      const wav = encodeWavBuffer([s[which]], s.resolved.sampleRate);
      res.writeHead(200, {
        'content-type': 'audio/wav',
        'content-disposition': `attachment; filename="${cfg.scenario}-${which}.wav"`,
      });
      return res.end(wav);
    }

    if (p === '/api/render-wav' && req.method === 'POST') {
      const body = await readBody(req);
      if (body.kind === 'session') {
        const s = runSessionSync(body.session || {});
        if (!s.ok) return sendJson(res, 422, s);
        const audio = body.path === 'iem' ? s.iemAudio : s.echoAudio;
        const wav = encodeWavBuffer([audio], s.resolved.sampleRate);
        res.writeHead(200, { 'content-type': 'audio/wav' });
        return res.end(wav);
      }
      const r = processMaterial(body.params || {});
      if (!r.ok) return sendJson(res, 422, { ok: false, message: r.design.message });
      const wav = encodeWavBuffer([r.output], r.sampleRate);
      res.writeHead(200, { 'content-type': 'audio/wav' });
      return res.end(wav);
    }

    if (p.startsWith('/api/')) {
      return sendJson(res, 404, { ok: false, message: '未知 API: ' + p });
    }

    if (req.method !== 'GET' && req.method !== 'HEAD') {
      res.writeHead(405);
      return res.end('Method Not Allowed');
    }
    return serveStatic(req, res, url);
  } catch (err) {
    return sendJson(res, 500, { ok: false, message: String(err && err.message || err) });
  }
});

server.listen(PORT, HOST, () => {
  console.log(`线性相位均衡器界面已启动:  http://localhost:${PORT}`);
  console.log(`自检接口:  http://localhost:${PORT}/api/verify`);
});
