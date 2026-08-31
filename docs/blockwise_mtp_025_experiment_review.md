# 2026-08-31 blockwise MTP 实验包复核

本次只复核已外传的报告和 collect 脱敏三件套，没有访问实验服务器或原始日志。
用户已进一步确认：当前运行是 **0.25rc1 / ARM / Python 3.12 / NPU / GLM-5.2 W8A8**，
在容器内切换了分支，重新编译了算子及 Mooncake 0.3.13。
本复核按此运行基线处理；历史采集器的包元数据不覆盖这项确认。提交和运行路径仍按各次实验分别列出。

## 当前能下的结论

**最新 MTP 尝试尚未越过 P 的启动门槛，不能据此认定 MTP 缓存映射或图执行失败。**
实验树也尚未包含本私仓的 blockwise MTP 映射补丁。

| 证据 | 实际范围 | 能证明什么 / 不能证明什么 |
| --- | --- | --- |
| 实验分支 `659291eca` | 父提交为 `d1bf0bad2`，新增 25 个报告文件，生产代码没有变化 | 能确认推送的源码基线；不能证明服务器没有额外 dirty 修改 |
| `LATEST.md`、issue 008 | 报告称 `d1bf0bad2`、MTP 关、DCP size=1、Decode `FULL_DECODE_ONLY`；smoke 通过、精度 5/5 | 是实验方的无 MTP 成功记录；本包没有对应晚间三件套或具体精度断言，不能扩写为完整精度保证 |
| issue 009，20:34 的尝试 1 | P1/D3，draft `enforce_eager=true`；P 的 ZMQ bind 报地址占用 | 最新已记录失败在 connector 启动期；“旧进程未清理”是候选原因，尚无监听者归属证据 |
| 15:31 三件套 | `baf3cbcf2` dirty、layerwise；D 有 6 类异常摘要 | 历史 planner/引擎退出证据，不是晚间 add_block MTP 失败栈 |
| 16:13 三件套 | `baf3cbcf2` dirty、layerwise，P/D/proxy 三侧 | 历史 layerwise 事件；异常样本为 0 不代表 blockwise MTP 通过 |

来源：[实验目录](https://gitcode.com/shichangzhang064/vllm-ascend/tree/exp%2Frebase25-add-block-20260831/docs/rebase25-exp)、
[最新结果](https://gitcode.com/shichangzhang064/vllm-ascend/blob/exp%2Frebase25-add-block-20260831/docs/rebase25-exp/LATEST.md)、
[MTP 尝试记录](https://gitcode.com/shichangzhang064/vllm-ascend/blob/exp%2Frebase25-add-block-20260831/docs/rebase25-exp/issues/009-add-block-mtp.md)。
读取时实验分支为 `659291ecafd5626ba8ff93a81babefa4f3d23fac`。
另一个 `mte_fuse_0723_mooncake_test_0827_add_block` 链接当时仍是 `d1bf0bad2`，
其 Git 树里没有 `docs/rebase25-exp`，报告实际新增在 `exp/` 分支。

## 1. 不能混在一起的三条线

- issue 001/002：旧的非 DSA offload Mooncake 路径，DCP8 下 `_get_kv_split_metadata`
  越界，无 MTP 也复现；它不能说明当前 DCP size=1 的 blockwise MTP 根因。
- issue 003–006 和 collect 三件套：旧 layerwise/fused planner 排障，树为
  `baf3cbcf2` 且 dirty。其临时 patch、SKIP 开关、shape 修补不能直接照搬到 `d1bf0bad2`。
- issue 007–009：切到 add_block，先完成无 MTP eager/graph，再尝试 P1/D3。
  最新公开失败是 P 端口绑定，不是旧 IndexError 或 planner Segfault。

`issues/ANALYSIS.md` 标题仍称 layerwise 是“当前主线”，已经滞后于 `LATEST.md` / 008 / 009。
旧的“不要整树切到 add_block”也是当时的历史计划，不是当前行动要求。

issue 008 同时改变了 offload 配方和共享内存清理，成功只证明组合可运行，
不能单独坐实之前挂起一定由 shm 引起。不要据此把清空 `/dev/shm` 变成常规重试操作。

## 2. 运行基线按用户确认的 rc1，区别于历史包元数据

用户已确认在容器中切分支、重新编译算子及 Mooncake，当前运行是 **rc1 + Mooncake 0.3.13**。
不再把版本混用列为当前已知故障，也不要求为此重新安装或重复确认版本。

历史三件套 `environment.packages.vllm=0.25.1` 仅是**当时采集器解释器**读到的安装包元数据，
其 scope 明确不是正在运行的 worker 导入证明，两个运行来源匹配字段均为 null，
后一轮命令还来自脚本 fallback。切换源码/重新编译与安装包元数据不同步可以并存；
这里不据旧字符串推翻用户确认的当前源码基线。

后续复现只需随新 attempt 记录实际 commit、dirty 状态、源码与构建产物是否匹配，
用于区分 rc1 上的具体改动。实际路径留在机内，仅返回匹配布尔值与 hash。
重编译是用户已完成的操作；现有失败栈也没有证据要求再次重编。

本次 ZMQ bind 地址占用与 Mooncake 0.3.13 或 W8A8 没有直接因果证据，
不要为此换 wheel、换量化或回滚 rc1。W8A8 仍不能替代 KV/Indexer dtype 与 scale 的布局信息。

## 3. 端口错误：落点确定，占用原因未定

对照 `d1bf0bad2` 的
[mooncake_connector.py](../vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py)，
worker 构造握手端口、`KVCacheSendingThread.run` 绑定监听的公式为：

```text
base = kv_port + data_parallel_rank * TP * PP * PCP
rank_offset = (pp_rank * PCP + pcp_rank) * TP + tp_rank
handshake_port = base + rank_offset
```

`zmq_ctx(ROUTER, ...)` 最终调用 socket bind；失败后线程退出，注册调用方抛出
`KV Cache sending/receiving thread failed to start`。后者是汇总错误，不是第二个独立根因。
这个握手端口与 `TransferEngine.get_rpc_port()` 返回的 TE RPC 端口也不是同一概念。

**公式不含 `num_speculative_tokens`，没有“每一个 MTP 步骤新占一个握手端口”的实现。**
所以 issue 009 的“MTP 多端口更易撞”应撤回；MTP 开启触发了重启，不等于它本身增加端口数。
如果某种配置导致重复构造/注册 connector，仍可能与 MTP 初始化有间接关系，现有材料无法排除。

另外，`KVCacheSendingThread started listening ...` 日志写在 bind **之前**；
不能拿这行日志单独证明绑定成功。`ready_event.set()` 在成功进入 socket 上下文之后。

### 需要区分的候选原因

| 候选 | 机内需要看到的证据 | 对应处理 |
| --- | --- | --- |
| 上一轮 worker/旧实验仍监听 | 新实例启动前端点已被占用，持有者属于旧尝试 | 仅停止已确认归属自己的旧实例；确认退出后重试 |
| 多服务共享 network namespace 且端口区间重叠 | 各实例计算端口一致，持有者属于另一当前实例 | 协调不重叠的 `kv_port` 区间，并同步 scheduler/worker/远端元数据配置 |
| DP rank 或 TP/PP/PCP 解析与预期不一致 | 实际 rank/size 组合产生重复端点 | 修启动拓扑/配置，不能靠随机换端口遮住 rank 错误 |
| 同一 worker 重复注册 | 同一匿名进程在同一尝试中两次进入 `register_kv_caches`，第二次撞自己的监听 | 才进入初始化生命周期修复；需要精简调用栈及注册次数 |

同一网络命名空间内不同绑定地址（包括通配地址）也可能相互覆盖，端口检查不能只按文本 IP 相等判断。
源码中注册函数没有针对“再次启动发送线程”的幂等保护，但这只是潜在风险；
当前 NPUModelRunner 的初始化路径可见一次 connector 注册，包内没有证明它被重复调用。
不应仅凭错误字符串就加“端口忙时自动跳过注册”或全局随机端口：这可能连接到旧实例、破坏缓存身份。

也不要 `pkill` 所有 vLLM、删他人 pod、清空全机 `/dev/shm`，或只清主进程不检查子 worker。
共享内存文件清理本身不能释放另一个活进程仍持有的 TCP 监听 socket。

## 4. 推送的实验分支尚不含我们的 MTP 补丁

实验分支的生产代码与 `d1bf0bad2` 相同，未包含
[`ea5db1de7`](https://github.com/lsjfy-open-com/vllm-ascend/commit/ea5db1de718aab49eb32dcc6935b53464d57b97c)。
那次补丁才增加 `dsa_cache_layout`、`dsa_block_group_ids`、按物理缓存身份匹配及覆盖范围校验。
本私仓 `codex/blockwise-mtp-025` 已包含这些变化和后续文档答疑。
这是对推送的 Git 树的判断，不代表已检查容器的额外 cherry-pick 或未提交补丁；
若容器另有改动，以本次实际 commit/差异为准，不能只凭分支名称判断补丁缺失。

“当前失败”既不能说明该补丁失效，也不能说明旧基线已经支持 P1/D3。
端口绑定发生在真正处理 PD 请求之前；缓存映射补丁不会替代端口清理或拓扑修正。

建议分两步，不混变量：先在当前失败提交上确定并解决端口问题，记录 P Ready 即止；
然后 P/D 同时切到包含 `ea5db1de7` 的同一受控提交，核对加载来源，再执行最小 MTP 验收。
无需为了端口问题重跑旧的 DCP8、layerwise 或完整精度矩阵。

交付场景既然是 P1/D3，后续沿用该组合：先两侧全局 eager，再恢复实际 target 图配置。
GLM 在该分支的 proposer 命中 `model_type` 的 `glm` 判断时强制 draft eager；
`speculative.enforce_eager=true` 也明确要求 draft eager。
因此 `FULL_DECODE_ONLY + MTP` 仍表示 target graph + draft eager，不是 draft graph 能力验收。
继续保持 `multistream_overlap_shared_expert=false`，不要关闭 D 侧必要的 offload `use_fused_overlap=true`。
详细通过标准沿用 [迁移文档](blockwise_mtp_025_migration.md)，不新增全排列实验。

## 5. collect 结果的缺口与最小补采任务

已有 collect 包做到了限制原始内容外传，但不是晚间 MTP 失败的对应包。
`analysis.md` 主要是自动模板，列出“待验证”，尚未完成引用事件编号的人工归因。
不能把“唯一异常样本 0”或 `matches_reviewed_experiment=true` 当成当前实验通过：
后者匹配的是旧采集器审阅的 `baf3cbcf2`。

15:31 的 E002 能看到 TP0 的 manager 异常链，随后有 EngineDead 类异常；
错误类型/原因被折叠成 `other_exception` / `message_withheld`。
issue 005 称当时是移植漏定义 `external_plan_debug` 的 NameError；这是报告的解释，
三件套本身不足以独立确认具体未定义变量。无需为了当前端口问题重做这轮历史排障。

旧采集器的白名单没有 `NameError`、`ZMQError` 和 `Address already in use` 原因码，
事件也以 layerwise 为主。继续使用原版本可能将本轮关键原因隐藏；
**不能因此放开输出任意异常原文或完整配置**。后续采集应增加固定错误类型/原因码、
blockwise 阶段和受限配置字段，并用含凭据、地址、prompt 的合成输入确认不会泄漏。
本次没有修改采集器，也不把下面字段描述为现有脚本已支持的选项。

### 给实验机执行者：先补一次启动包，暂不压测

1. 为这次重试建立新的匿名 attempt，所有 P/D/proxy 使用同一个关联范围。
   只选本次进程启动到 Ready 或首次退出的日志范围，禁止拼入 15:31/16:13 的旧日志。
2. 在停服或改配置前，机内确认报错端点的监听者；原始 `ss/lsof`、进程 argv、容器详情只交给
   本地确定性过滤程序，不输出到外部 agent 上下文。未知所有者就标记 unknown，不自动终止。
3. 输出下面白名单表。数值端口、PID、网络地址、完整路径和请求内容不需要外传；
   使用本次尝试内一致的主机/实例/进程匿名别名，仅保留关系、计数和 rank。
4. 区分清理前和清理后；只处理确认属于本实验的实例。先确认 P 能 Ready；失败时停止堆叠开关。
5. 拿到 Ready 后再执行上一节的补丁部署和小请求集。额外 group 等显式不支持布局出现时停止，
   回传布局计数，不靠删校验继续跑。

| 输出组 | 仅需返回的字段 |
| --- | --- |
| 运行身份 | 角色、匿名实例、attempt；Ascend/vLLM commit、dirty 布尔、包版本；worker 导入与目标 checkout 匹配布尔；未观测项为 null |
| 有效配置 | 实际 connector 类、`dsa_pd_offload`、Main/Indexer group 数；DP/TP/PP/PCP/DCP size 和 rank；P/D speculative 步数、全局/局部 eager、实际 graph mode |
| 绑定复核 | 匿名 namespace/端点别名；期望端点总数、唯一数、重复数；启动前已有监听数量；所有者为本次/旧尝试/其他/未知；同一匿名进程注册次数（可观测才填） |
| 首个失败 | 固定异常类型 `ZMQError`；固定原因 `address_in_use`（实际命中才填）；仓库相对文件/已审核函数名；角色/rank；匿名端点；不要异常原文 |
| 生命周期 | bind 成功数、worker Ready 数、API Ready 布尔；前后窗口顺序；缺失/截断计数；只看到 bind 前的 started-listening 日志不能记 bind 成功 |
| 后续功能小测 | P/D 同提交匹配；真实 PD 命中数、必要 MTP Main/Indexer 组件计数；各 rank 结果种类计数；生成/接受/拒绝候选计数；准确性检查的断言类型与通过数，不含输出文本/token IDs |

绑定失败的当前阻塞无需 checkpoint 内容、请求全文、完整日志或更多性能数据。
诊断未知值保持 null；数据不支持的“残留进程已坐实”“MTP 功能错误”“rc1 上 MTP 功能已通过”都不要写。
“运行版本是 rc1”已经由用户确认，与“MTP 功能验收通过”是不同结论。

## 本次复核交付范围

只新增此分析文档，没有修改实验分支、运行代码、端口分配、采集器或启动脚本。
已核对 Git 树差异、当前 connector/proposer 源码与四套 facts 的证据边界。
未在 NPU 上复现或修复；没有新增实机测试结论。
