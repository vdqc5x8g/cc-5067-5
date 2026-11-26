// 命令行验证：node scripts/verify.mjs
// 运行与界面/API 相同的客观自检，并把两条路径的真实 WAV 与报告写到 output/。

import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { runVerification } from '../public/dsp/verify.js';
import { runSessionSync } from '../public/dsp/session.js';
import { designFir } from '../public/dsp/fir-design.js';
import { convolveDirect } from '../public/dsp/convolution.js';
import { generateSweep, generateDrumLoop } from '../public/dsp/audio.js';
import { encodeWavBytes } from '../public/dsp/wav.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const outDir = join(__dirname, '..', 'output');
mkdirSync(outDir, { recursive: true });

const report = runVerification();
console.log(`\n线性相位 EQ 自检：${report.passed}/${report.total} 通过\n`);
for (const r of report.results) {
  console.log(`${r.pass ? 'PASS' : 'FAIL'}  ${r.title}${r.detail ? '\n        ' + r.detail : ''}`);
}

function writeWav(name, bytes) {
  const p = join(outDir, name);
  writeFileSync(p, Buffer.from(bytes));
  console.log('写出', p, bytes.length, '字节');
}

// 默认配置的双路径（旧尾情景）—— 与界面初始状态、/api/demo 完全一致
const session = runSessionSync({ scenario: 'old-tail' });
writeWav('session-oldtail-echo.wav', encodeWavBytes([session.echoAudio], 48000));
writeWav('session-oldtail-iem.wav', encodeWavBytes([session.iemAudio], 48000));

const sessionB = runSessionSync({ scenario: 'new-effective' });
writeWav('session-newhit-echo.wav', encodeWavBytes([sessionB.echoAudio], 48000));
writeWav('session-newhit-iem.wav', encodeWavBytes([sessionB.iemAudio], 48000));

// 主链路素材样例：脉冲、扫频、鼓循环经当前 EQ 核的真实卷积
const points = [
  { f: 0, db: 4.5 },
  { f: 0.08, db: 4.5 },
  { f: 0.22, db: 1.2 },
  { f: 0.45, db: 0 },
  { f: 0.7, db: -1.5 },
  { f: 1, db: -1.5 },
];
const design = designFir({ points, N: 255, window: 'hann', sampleRate: 48000 });

const impulse = new Float64Array(4800);
impulse[0] = 1;
writeWav('processed-impulse.wav', encodeWavBytes([convolveDirect(impulse, design.kernel)], 48000));

const sweep = generateSweep({ sampleRate: 48000, durationSec: 3 });
writeWav('processed-sweep.wav', encodeWavBytes([convolveDirect(sweep, design.kernel)], 48000));

const drums = generateDrumLoop({ sampleRate: 48000, bpm: 124, bars: 2 });
writeWav('processed-drums.wav', encodeWavBytes([convolveDirect(drums, design.kernel)], 48000));

writeFileSync(
  join(outDir, 'verify-report.json'),
  JSON.stringify(
    {
      ...report,
      artifacts: {
        oldTail: {
          echoPeakFile: session.echoMeasure.peakSampleFile,
          iemPeakFile: session.iemMeasure.peakSampleFile,
          difference: session.difference,
          comb: session.comb,
        },
      },
    },
    null,
    2
  )
);

console.log('\n峰位/拍差（旧尾默认）:', JSON.stringify(session.difference, null, 2));
if (!report.pass) process.exit(1);
