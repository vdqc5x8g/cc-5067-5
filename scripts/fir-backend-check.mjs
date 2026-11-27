// 命令行仅作辅助：图形入口是 npm start 后的“⑤ 实时试听 / 输出文件”面板。
// 此脚本生成确定性信号，不读取 output/ 或任何手工修改的数据。
import { verifyFirBackend, BACKEND_BUILD_LOCK } from '../public/dsp/fir-backend.js';

const long = process.argv.includes('--short') ? false : true;
const evidence = verifyFirBackend({ long });
const passed = evidence.checks.filter((c) => c.pass).length;

console.log(`\nFIR 后端适配检查：${passed}/${evidence.checks.length} 通过（${long ? '长滤波器' : '快速'}）`);
console.log('构建锁定:', JSON.stringify(BACKEND_BUILD_LOCK));
for (const c of evidence.checks) {
  const detail = c.maxAbs != null
    ? `maxErr=${c.maxAbs.toExponential(3)} rms=${c.rms?.toExponential?.(3)} len=${c.lengthActual}/${c.lengthExpected}`
    : c.id.includes('tail-once')
      ? `beats=${c.beats?.actualBeatCount}/${c.beats?.expectedBeatCount} alignedErr=${c.originalDifference?.alignedMaxAbs?.toExponential?.(3)}`
      : c.id;
  console.log(`${c.pass ? 'PASS' : 'FAIL'}  ${c.id}  ${detail}`);
}
if (!evidence.pass) process.exit(1);
