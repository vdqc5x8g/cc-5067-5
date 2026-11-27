# 线性相位 FIR 均衡器

浏览器图形界面、AudioWorklet 实时卷积和 Node.js 离线验证共用同一套线性相位 FIR 契约。当前构建锁定为：

- 应用版本：`1.1.0`
- FIR 后端 API：`2.0.0`
- 卷积后端：`overlap-add-js-2.0.0`
- lockfile：`package-lock.json` `lockfileVersion: 3`
- 运行时第三方依赖：0；开发依赖仅 `jsdom`

## 图形入口

```bash
npm ci
npm start
```

打开 <http://localhost:3000>。命令行检查只是辅助，线性相位的主要操作和判定都在页面中完成。

1. 在 **“⑤ 实时试听 / 输出文件”** 面板设计 FIR、拖动 EQ 曲线并试听或渲染输出。
2. 拖动 EQ 曲线后，页面会自动重新设计并校验后端；也可以点击 **“校验响应 / 延迟音频”**（`#btn-validate-backend`）立即执行。
3. 点击 **“锁定对比 A/B”** 后，会保留 A/B 状态并立即校验滤波响应、延迟峰位、相位和尾部。
4. 在 **“⑥ 母带回印 × 固定时钟耳返”** 面板进行母带试听；该面板显示同一个线性相位新后端状态。
5. 页面状态标识：
   - `#linear-phase-status` / `[data-testid="linear-phase-status"]`：线性相位新路径运行状态。
   - `#master-linear-status`：母带试听面板的新后端状态。
   - `#backend-status`：FIR 后端校验状态。
   - `#response-anomaly`：滤波响应异常；正常时为绿色，异常时为红色并给出原因。
   - `#delay-anomaly`：延迟音频异常；覆盖群延迟、冲激峰位、相位极性和尾部重复。
   - `#backend-evidence`：最近一次真实卷积执行证据。
   - `window.__backendEvidence`：供自动化或控制台读取的结构化证据。

原有 EQ 拖拽、A/B 锁定、会话、播放、导出等交互保持不变。

## 卷积尾部与延迟契约

适配器同时兼容两种尾部冲洗约定：

- `deferred-drain`：`end()` 取最后有效块，随后通过 `drain()` / `drain(Infinity)` 输出长分区尾部。
- `merged-final`：`finish()`、`end()` 或 `flush()` 在最后有效块中合并返回尾部，之后不再重复 drain。

两个参考后端都报告相同的整块延迟：

```text
wholeBlockLatencySamples = blockSize
```

该值是分块延迟报告，不替代 FIR 的算法群延迟。I 型奇数长核的标称群延迟为整数 `(N - 1) / 2`；II 型偶数长核为半整数。适配器允许保留旧时标，也允许通过真实冲激重新测量峰位/亚样本群延迟。

离线完整输出严格保持：

```text
output.length = input.length + kernel.length - 1
```

算法延迟和 `N - 1` 个尾部样本不会通过文件裁切隐藏。最终音轨可与原轨对齐做差；末块反相抵消探针和尾拍计数用于发现尾部被记录两次的问题。

## 边录边换核

录音或实时处理中更换核时，新核从下一个 128 样本 Worklet quantum 生效。旧 overlap-add 通道不会被立即清空，而是转入 drain 列表；旧核尾部与新核当前输出逐样本相加。验证会对照确定性的分段直接卷积，检查冲激位置、相位、极性和 Float32 误差。

## 精度与容差

所有误差阈值均显式声明，不把不同精度后端混入同一容差：

| profile | 最大绝对误差 | RMS | 相位相干系数 |
| --- | ---: | ---: | ---: |
| `float64` | `1e-8` | `1e-9` | `0.999999` |
| `float32` | `2e-5` | `5e-6` | `0.9999` |
| `wasm` | `5e-5` | `1e-5` | `0.999` |

Worklet 的 Float32 热切换证据按 `float32` 容差判定；Node Float64 离线路径按更严格的 `float64` 容差判定。

## 旧配置迁移

旧版 `version: 1` 参数、localStorage 会话或导出的参数文件仍可载入。`migrateFilterConfig()` 会升级到 schema v2，保留 EQ 点、滤波器长度、场景和会话数据，并明确记录：

```text
cropSamples = 0
```

迁移后的检查会确定性生成 60000 样本长音频并真实执行 FIR 卷积，不依赖 `output/` 中的手工文件。

## 测试与辅助命令

```bash
npm test
npm run backend:check
npm run verify
```

- `npm test`：Node 测试、DSP 契约、双尾部后端、热换核、迁移和 jsdom UI 冒烟。
- `npm run backend:check`：FIR 后端长证据命令行辅助检查。
- `npm run verify`：页面/API 同一套 DSP 自检的命令行辅助入口。
- `npm run smoke:browser`：可选的真实浏览器冒烟（需要本机可用的自动化浏览器环境）。

## 干净目录验证

FIR 后端检查只使用测试内确定性信号和仓库内源码。可在净目录中执行：

```bash
clean=$(mktemp -d /tmp/fir-clean-XXXXXX)
tar --exclude='./node_modules' --exclude='./.git' --exclude='./output' \
  -cf - -C . . | tar -xf - -C "$clean"
cd "$clean"
npm ci
npm run backend:check
npm test
```

上面的命令复制当前完整工作树到临时净目录，并排除 `node_modules`、`.git` 和生成的 `output/`。检查群延迟、卷积误差、尾部排空和热换核时只使用仓库源码与测试内确定性信号，不读取或修改手工准备的数据文件。若验证已提交版本，也可改用 `git archive HEAD | tar -x -C "$clean"`。
