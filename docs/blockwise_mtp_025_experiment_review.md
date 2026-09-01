# 2026-08-31 blockwise MTP 实验包复核

本次只复核已外传的报告和 collect 脱敏三件套，没有访问实验服务器或原始日志。
用户已进一步确认：当前运行是 **0.25rc1 / ARM / Python 3.12 / NPU / GLM-5.2 W8A8**，
在容器内切换了分支，重新编译了算子及 Mooncake 0.3.13。
本复核按此运行基线处理；历史采集器的包元数据不覆盖这项确认。提交和运行路径仍按各次实验分别列出。

## 当前能下的结论

> **2026-09-01 更新：原分支已天然跑通当前目标冒烟。**
> 最新实验包记录 `d1bf0bad2`、`dirty=false`，没有合入本私仓 MTP 映射补丁；
> P=`num_speculative_tokens=1`、D=`3`、两侧 draft eager、Decode target
> `FULL_DECODE_ONLY`、DCP size=1、fused offload、standard blockwise proxy，
> 在正确清理孤儿进程后 P/D Ready，32-token smoke 返回 `TEST_PD_OK`。
> 因此本补丁不是这套配置跑通的前置条件，不应为该已通过场景继续改 connector。

**最新 MTP 尝试已在原 `d1bf0bad2` 上越过启动和请求冒烟门槛。**
实验树不含本私仓的 blockwise MTP 映射补丁，这恰好证明当前 P1/D3 配置不依赖该补丁。

| 证据 | 实际范围 | 能证明什么 / 不能证明什么 |
| --- | --- | --- |
| 实验分支 `32971d499` | 生产代码父提交为 `d1bf0bad2`；后续提交只新增/更新报告与实验脚本 | 最新三侧 facts 记录 `head=d1bf0bad2`、`dirty=false` |
| `LATEST.md`、issue 008 | 报告称 `d1bf0bad2`、MTP 关、DCP size=1、Decode `FULL_DECODE_ONLY`；smoke 通过、精度 5/5 | 是实验方的无 MTP 成功记录；本包没有对应晚间三件套或具体精度断言，不能扩写为完整精度保证 |
| issue 009，20:34 的尝试 1 | P1/D3，draft `enforce_eager=true`；P 的 ZMQ bind 报地址占用 | 历史失败；后续已确认错误停服方式留下孤儿进程 |
| issue 009，21:24–21:38 的尝试 4 | 同一生产代码 `d1bf0bad2`，正确停服后 P/D Ready；MTP P1/D3 + target FULL 图，32-token `TEST_PD_OK` | 证明当前目标配置可起服、可完成一次生成冒烟；尚不等于 MTP 精度、接受率、并发和长稳验证 |
| 15:31 三件套 | `baf3cbcf2` dirty、layerwise；D 有 6 类异常摘要 | 历史 planner/引擎退出证据，不是晚间 add_block MTP 失败栈 |
| 16:13 三件套 | `baf3cbcf2` dirty、layerwise，P/D/proxy 三侧 | 历史 layerwise 事件；异常样本为 0 不代表 blockwise MTP 通过 |

来源：[实验目录](https://gitcode.com/shichangzhang064/vllm-ascend/tree/exp%2Frebase25-add-block-20260831/docs/rebase25-exp)、
[最新结果](https://gitcode.com/shichangzhang064/vllm-ascend/blob/exp%2Frebase25-add-block-20260831/docs/rebase25-exp/LATEST.md)、
[MTP 尝试记录](https://gitcode.com/shichangzhang064/vllm-ascend/blob/exp%2Frebase25-add-block-20260831/docs/rebase25-exp/issues/009-add-block-mtp.md)。
本次复核读取到实验分支 `32971d499179ae3294e4ce09a842bef2d286e313`。
另一个 `mte_fuse_0723_mooncake_test_0827_add_block` 链接当时仍是 `d1bf0bad2`，
其 Git 树里没有 `docs/rebase25-exp`，报告实际新增在 `exp/` 分支。

## 1. 不能混在一起的三条线

- issue 001/002：旧的非 DSA offload Mooncake 路径，DCP8 下 `_get_kv_split_metadata`
  越界，无 MTP 也复现；它不能说明当前 DCP size=1 的 blockwise MTP 根因。
- issue 003–006 和 collect 三件套：旧 layerwise/fused planner 排障，树为
  `baf3cbcf2` 且 dirty。其临时 patch、SKIP 开关、shape 修补不能直接照搬到 `d1bf0bad2`。
- issue 007–009：切到 add_block，先完成无 MTP eager/graph，再尝试 P1/D3。
  最新公开结果已经在正确停服后跑通，不是旧 IndexError 或 planner Segfault。

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

最新 21:39 facts 已记录 `d1bf0bad2` 且 dirty=false；后续验证只需随新 attempt 继续记录实际 commit、dirty 状态、源码与构建产物是否匹配，
用于区分 rc1 上的具体改动。实际路径留在机内，仅返回匹配布尔值与 hash。
重编译是用户已完成的操作；现有结果没有证据要求再次重编。

后续成功重试进一步支持：此前 ZMQ bind 地址占用与 Mooncake 0.3.13、MTP 语义或 W8A8 没有直接因果证据，
不要为此换 wheel、换量化或回滚 rc1。W8A8 仍不能替代 KV/Indexer dtype 与 scale 的布局信息。

## 3. 历史端口错误：已确认是错误停服留下孤儿进程

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
这个握手端口与 `TransferEngine.get_rpc_port()` 返回的 TransferEngine RPC 端口也不是同一概念。

**公式不含 `num_speculative_tokens`，没有“每一个 MTP 步骤新占一个握手端口”的实现。**
所以 issue 009 的“MTP 多端口更易撞”应撤回；MTP 开启触发了重启，不等于它本身增加端口数。
最新 stopfix 结果确认当时 pod 内错误调用宿主机脚本，旧 worker/孤儿进程没有退出；
在宿主机正确执行停服后，同一 MTP 配置成功。无需再把重复注册或拓扑重叠列为当前首要嫌疑。

另外，`KVCacheSendingThread started listening ...` 日志写在 bind **之前**；
不能拿这行日志单独证明绑定成功。`ready_event.set()` 在成功进入 socket 上下文之后。

### 这次的处理边界

报告已经给出本次占用者属于 p1/d0/d1 的孤儿 VLLM，并通过正确停服后的同配置成功重试形成闭环。
仍不能将处理方式泛化成全局清理：只停止确认属于本实验的 worker；不要随机改端口、
`pkill` 所有服务、删除他人 pod，或把清空 `/dev/shm` 当成释放 TCP socket 的手段。
以后若相同错误在正常停服后复现，再重新区分端口区间重叠、rank 配置和重复注册。

## 4. 原分支跑通后，对本私仓补丁的重新定位

最新实验提交 `32971d499` 只增加报告和实验脚本；其生产代码父提交仍是 `d1bf0bad2`。
21:39 三侧 facts 均记录 Ascend `head=d1bf0bad2`、`dirty=false`；启动脚本明确向 P/D 分别传入
MTP 1/3 步和 `enforce_eager=true`，D 还传 `FULL_DECODE_ONLY`。重试脚本实际导出
`ENABLE_MTP=1`，而不是只在文档里声明。D 的三件套记录两次 graph capture complete 和 API Ready。

因此应撤回“必须先合入 `ea5db1de7` 才能验证 P1/D3”的方向：

- 对当前 GLM-5.2、P1/D3、DCP size=1、现有 manager group 布局，原分支已经具备能力；
- `ea5db1de7` 的物理身份描述符、显式 block 组号和覆盖检查没有参与本次成功；
- 在没有复现错层、漏传、跨 P/D 组顺序不一致之前，不能把这些防御性变化作为 bug fix 合入；
- 新协议增加握手字段、校验和兼容面，本身也有回归风险。最稳妥的处理是保留实验分支不动，
  将 `codex/blockwise-mtp-025` 作为研究分支，先不部署、不宣称必要。

这次 `TEST_PD_OK` 的脚本判据是 HTTP JSON 有非空生成内容，并非只检查 200；
但脚本会把输出内容写入实验机原始日志，不能外传。当前材料没有 MTP 候选/接受/拒绝统计，
所以“配置已打开并完成端到端冒烟”成立，“MTP 一定贡献了 token/加速”尚未由统计证明。
同样，MTP 关的精度 5/5 与 eager 压测 12/12 不能转写成 MTP 开的精度/压测结果。

下一步无需先改 connector。若目标只是确认功能，再在原 `d1bf0bad2` 上补一次 MTP 开的
固定精度小集及 speculative 汇总统计；只有出现确定的错层/缺组件/容量截断，再用具体失败输入
评估 `ea5db1de7` 中哪一部分值得最小化迁移。DCP>1、额外 manager group、不同模型布局
仍未由这次冒烟覆盖，不能将“天然跑通”外推到这些拓扑。

## 5. 仍需补的证据很小

21:39 的 collect 包已经对应成功重试：P、D 和 proxy 均记录 `d1bf0bad2`、dirty=false 和 API Ready；
D 还记录两次 graph capture complete。它足以辅助确认服务启动与图捕获，原始输出无需外传。
但采集器没有记录 blockwise MTP 候选/接受/拒绝统计，也没有证明候选实际贡献了生成 token。

下一轮无需再采端口排障包，也无需先部署本私仓补丁。仅在原分支 MTP 开的精度小测中增加白名单汇总：

- 实际 connector、P/D speculative 步数、target graph mode 及 draft eager；
- speculative 候选、接受、拒绝和回退的汇总计数；
- 固定精度断言的通过/失败数和首次分歧位置，不含 prompt、输出文本或 token IDs；
- PD 命中、Main/Indexer 传输阶段和各 TP result kind 的计数，不含 block IDs、地址和请求标识。

这些结果能区分“配置并成功生成”与“MTP 真正参与且结果正确”。只有出现具体的缓存错层或漏传，
再提取布局的层数、group 数、dtype/shape 和固定错误码，评估是否需要最小 connector 修复。

## 6. 最新精度/简单压测与 PCP/DCP 支持判断

2026-09-01 的实验分支更新到 `a104e91a0`；生产代码仍是 `d1bf0bad2`，新增的是结果、
性能基线和复现实验脚本。MTP+图配置新增两项正向结果：

| 项目 | 结果 | 结论边界 |
| --- | --- | --- |
| 精度小集 | `MAX_TOKENS=256`，5/5 `ACCURACY_PD_OK` | 当前固定小集通过，不代表完整模型评测 |
| 简单压测 | concurrency=4，N=16，input≈512，max_tokens=32；16/16 成功 | 证明短负载并发可用，不是长稳或容量结论 |
| TTFT | p50 1.201s，mean 1.160s | 客户端首 token，包含 P、KV 传输和 D |
| E2E | p50 2.444s，mean 2.342s | 当前短输出工作负载的端到端时间 |

报告同时列出 eager/无 MTP 的 12 请求基线：TTFT p50 1.710s、E2E p50 11.561s。
两轮样本数不同，而且同时改变 graph 与 MTP，不能把差值单独归因给 MTP 或 graph；
可以将 MTP+FULL 图的数值作为当前整体配方基线。仍缺 speculative 接受/拒绝计数和更长负载。

### PCP 和 DCP 的角色

概念上，PCP 用于 P 的长 prompt 计算分片，DCP 用于 D 的 KV 序列分片；实际拆分配置通常是：

```text
P 服务：prefill_context_parallel_size = P_PCP (>1 才算开启)
        decode_context_parallel_size = 1
D 服务：prefill_context_parallel_size = 1
        decode_context_parallel_size = D_DCP (>1 才算开启)
```

PCP 会增加 Prefill 的执行 ranks；DCP 复用 TP ranks。`additional_config.enable_dsa_cp` 是另一条
DSA-CP 机制，不能与 PCP 或 blockwise PD connector 的 DSA 名称混为一谈。

shichang 当前实验脚本并没有打开 CP：P/D 都传 `--prefill-context-parallel-size 1`，
并把名为 `PREFILL_DCP_SIZE` / `DECODE_DCP_SIZE` 的变量作为
`--decode-context-parallel-size 1` 传入。**size=1 是单 rank，即关闭；“对称 DCP=1”不是开了 DCP。**
P 脚本中的 `PREFILL_DCP_SIZE` 只是变量名，不是 PCP。D 脚本还设置 `enable_dsa_cp=false`。

### 当前 blockwise DSA 代码为何不支持 CP>1

`MooncakeConnector` 在 `dsa_pd_offload=true` 且角色为 Decode consumer 时明确校验：

```python
if dcp_size * pcp_size != 1:
    raise ValueError("Blockwise DSA Decode requires DCP * PCP == 1 ...")
```

因此 D 服务把 DCP 或 PCP 设为大于 1 会在 connector 初始化时直接失败。P 侧没有这条同位置断言，
但也不能据此声明 PCP 可用：

- `_DsaParallelTopology` 只描述 TP/DP/PP，没有 PCP/DCP；
- DSA 的 `RemoteSource.endpoints_by_prefill_rank` 只按 Prefill TP size 构造；
- `_dispatch_dsa_commands` 用 Decode TP rank 映射一个 Prefill TP leader，未包含 PCP rank；
- DSA receiver 从 `DsaConnectorMetadata` 提前进入 `_start_dsa_commands`，不会走普通
  `MooncakeConnectorMetadata` 的 `_get_kv_split_metadata` CP 分片逻辑。

文件里 `remote_pcp_size` / `remote_dcp_size`、`_get_kv_split_metadata` 和 CP group-pull 代码
属于普通 Mooncake Pull 路径。它们存在不代表 blockwise DSA 的 Indexer D2D + Main D2RH 已支持 CP。

**当前支持声明应保持为 P_PCP=1、D_DCP=1。** 若要支持“P 开 PCP、D 开 DCP”，
需要先扩 DSA 协议和拓扑：携带 P/D CP size/rank，按 TP×PCP 发布全部 P endpoint，
明确 virtual block 到各 CP shard 的 Main/Indexer block 路由，D 各 rank 聚合完成结果，
并重新验证共享 Main owner、Indexer replicated/sharded 语义、MTP、prefix cache 和 graph。
仅删除 `DCP * PCP == 1` 校验会把不完整 metadata 送入错误 rank，不能作为实现。

## 本次复核交付范围

本次根据最新实验包更新此分析文档，没有修改实验分支、运行代码、端口分配、采集器或启动脚本。
已核对 Git 树差异、实际归档脚本、最新三侧 facts 和 `TEST_PD_OK` 判据。
实机成功是实验方提供的结果；本地没有 NPU，未独立复现。
