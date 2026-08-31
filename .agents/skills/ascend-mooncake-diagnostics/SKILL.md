---
name: ascend-mooncake-diagnostics
description: 在 Ascend NPU 实验机离线提取 vLLM Ascend Mooncake layerwise PD 启动或首请求故障的脱敏证据，区分 proxy、cache 布局和运行配置；仅回传关键结构化内容，不读取原始日志到 agent 上下文，不自动修复或重启服务。
---

# Mooncake PD 关键证据采集

按此文件执行，不凭经验跳步。脚本只需要 Python 3.12 标准库，不安装依赖、不访问外网、不加载模型、不初始化 NPU。

## 本次已知条件

- **本轮测试基线是 0.25rc1**，由用户确认。分别记录实际安装的 vLLM、vLLM Ascend 版本及各自 commit；不要自动补版本号，也不要混入此前 0.23 的测试、CI 或兼容性结论。
- 实验代码：`mte_fuse_0723_mooncake_test_0827`；审查基点 `baf3cbcf22851bdb97102ce477329bd9f621240e`。采到其他 commit 时报告差异，不擅自回退。
- 模型：GLM-5.2 W8A8；`multistream_overlap_shared_expert=false`。此前接近成功的实验关闭 MTP、开启 MC2，但必须收集实际 MC2 参数，不能把“MC2 开”转换成某个猜测的 flag。
- 目标为 Linux ARM64、NPU、CPython 3.12；用户已安装 Mooncake 0.3.13，仍需核实服务环境中的发行包。
- 昨天误用了普通 proxy。旧 PD 请求测试不能证明 layerwise 链路可用；也不能把请求到达前的初始化异常归因于 proxy。
- W8A8 是权重量化。FA、KV C8、Sparse SFA C8、LI C8 必须分别取证。GLM-5.2 的 shared Indexer 层不一定有独立 Indexer tensor。
- 此 skill 从私仓 main 分出独立诊断分支，仅用于分发工具；main 的代码版本不构成本轮实验基线。**不要把实验服务切到诊断分支，不要 cherry-pick 其他服务代码。**

## 数据边界：先由脚本读取，再由 agent 分析

原始日志、启动脚本、`config.json`、`quant_model_description.json`、`/proc/PID/cmdline` 只能由本地脚本读取。不得用 `cat`、`tail`、`rg -A/-B`、IDE 打开、终端回显等方式把其内容送入 agent 上下文。这条限制适用于工具输出，不只是最终回复。

不得运行 `env`、`ps aux`、全量 `pip freeze`，不得打印 HTTP 请求/响应、prompt、token IDs、凭证、真实地址和指针。不要开启全局 DEBUG：当前 layerwise proxy 的 DEBUG 分支会记录 `req_data`。

`collect.py` 使用允许字段重新生成材料，不做“整行替换几个秘密后回传”。未知异常消息、路径和函数名会被省略或归类为 `other_*`。不要为了补全内容而绕过过滤器；缺失信息进入下一轮采集清单。

只能回传每个角色导出目录内的 `facts.json`、`evidence.txt`、`analysis.md`。不得回传整个工作目录、manifest、correlation key、原始日志、模型文件或压缩它们形成的附件。不得把采集结果加入 Git。

## 第一步：锁定同一次实验

1. 获取实验机上 P、D、proxy 当前这一次启动对应的日志文件路径、实验代码目录、服务 Python 路径。只问路径，不让用户粘贴日志。历史日志与修正 proxy 后的新实验分别采集。
2. 用服务所在容器、同一个 Python/venv 执行脚本。不要在宿主机另一个环境采集后声称是服务版本。只读采集，不停止已有进程。
3. 能确定服务主进程 PID 就使用 PID；进程已退出则提供实际启动脚本路径。多节点每个角色分别执行，P 多实例使用不同 `instance`。每份 manifest 最多 16 个日志；只填同一实例、同一次启动，日志轮转按旧到新列出，不把同一内容重复列入。
4. 不知道路径、无法进入相同容器、输入读取失败，就报告缺项并停在本步。禁止扫描并打印整个日志目录内容。

## 第二步：在仓库外初始化私有工作目录

以下变量由实验机实际路径替换，不猜路径。`DIAG_PYTHON` 必须是服务 Python；`DIAG_SCRIPT` 是已经取得的本 skill 的 `scripts/collect.py`。

```bash
DIAG_PYTHON=/actual/service/venv/bin/python3
DIAG_SCRIPT=/actual/diagnostic-tools/ascend-mooncake-diagnostics/scripts/collect.py
DIAG_LOCAL=/actual/non-git/private-diagnostics/p-attempt-01
"$DIAG_PYTHON" "$DIAG_SCRIPT" init --work-dir "$DIAG_LOCAL"
```

`DIAG_LOCAL` 必须原先不存在、不在 Git 仓库内、不经过符号链接。脚本生成 `manifest.json` 和权限为 0600 的 `correlation.key`；后者不是业务密钥，仅用于同一实验的请求标识匿名化。

按 [采集字段说明](references/collection-contract.md) 修改 manifest 中的路径、角色和实例编号。不能读取输入日志/模型文件的原文来填写 manifest。`null` 表示未知，不填猜测值。

同一实验的 P、D、proxy 应在实验网内安全复制并使用**同一份** correlation key；不要各自使用新生成的 key。跨主机传输只在实验网内由已有获准方式完成，key 不出实验网。新一轮实验换新 key。不能共享 key 时照常各自采集，但注明请求无法跨包关联。

## 第三步：运行采集，检查覆盖范围

```bash
"$DIAG_PYTHON" "$DIAG_SCRIPT" collect \
  --manifest "$DIAG_LOCAL/manifest.json" \
  --output "$DIAG_LOCAL/export"
```

只允许 agent 读取 `export/` 下的三个文件。stdout 仅有固定状态，不包含输入。重复采集时选择新的输出目录；不得删除旧包来掩盖差异。

- 先看 `coverage`：截断、超长行、文件读取期间变化都必须报告。运行中的日志可能变化，这不是自动失败，但不得据此宣称异常不存在。
- 默认每份日志扫描前 128 MiB，可在 manifest 内增至最多 512 MiB；支持本地 `.gz`。到达上限不能推断余下日志没有错误。只处理对应尝试的文件，必要时让机房操作者在本地按该次尝试另存日志后再采集，禁止把片段回显给 agent。
- 检查 `errors_over_budget`、`events_over_budget`、`ambiguous_unprefixed_trace_lines`、`incomplete`。异常按结构去重并保留 rank 列表，不能只读 TP0 的文件。
- `facts.environment` 是执行采集器的 Python 环境；与 PID 的可执行文件匹配也不证明该进程加载的每个 Python 模块路径相同。
- `command.values` 是启动参数，`checkpoint.derived_metadata_flags` 是文件推导，只有 `runtime_observations` 才是已有的结构化运行时观测。没有日志就保持未知。

## 第四步：依据证据分析，不先入为主

完整读取 [分析决策与本分支陷阱](references/analysis-guide.md)。使用 [回传模板](references/report-template.md) 补充 `analysis.md`，只引用已导出的允许字段。证据编号必须加角色和实例，例如 `P0:E001`、`D0:T004`，不能跨包混用。

每个结论分为“观察到的事实 / 待验证假设 / 缺失证据 / 下一步”。不得把 mock 复现、静态审查或旧版本 CI 当成本次实机验证。没有找到错误时也不能写“全部通过”。

采集器不发请求、不重启、不修改环境、不上传结果。需要修正 proxy 或重新实验时，先提交当前摘要和最小实验建议。只有操作者明确批准该轮实验，才进入分析指南中的单请求步骤；不要同时改变 MTP、MC2、量化或 graph 配置。

## 交付

回传三个生成文件及基于它们的分析即可。上传动作由操作者按实验网规则执行，agent 不自行上传到 GitHub、聊天、网盘或其他外部服务。

完整 skill、脚本和合成测试在诊断分支；合成测试不需要 NPU：

```bash
python3 -m unittest tests.ut.tools.test_mooncake_diagnostics -v
```

只复制了 skill 目录时没有仓库 UT 文件，不要为运行测试切换实验服务代码。操作者可直接使用 [Cursor 任务入口](references/cursor-task.md)。
