# 按故障阶段分析

先看同一尝试的 P、D、PROXY 包是否齐全，以及 `correlation_scope` 是否相同。不同尝试的证据不能拼成一次成功链路。证据 ID 必须加角色和实例前缀。

本轮用户确认使用 **0.25rc1**。先核对每份包的实际 vLLM / vLLM Ascend 安装版本和源码 commit，再分析栈；此前 0.23 的 mock、CI、代码审查结果不构成本轮实机证据。若安装版本不同或无法核实，明确标为版本差异或未知，不自动改环境。

## 1. 先定位初始化还是请求故障

分别列出：加载权重、profile、KV 分配/注册、graph capture、API ready、首个请求、首个异常。只填观察到的阶段；P API ready 不等于每个 worker 的所有功能已经验证。

| 观察 | 可以说什么 | 不可以说什么 |
| --- | --- | --- |
| 异常栈在加载/注册函数，且有证据证明未发请求 | 优先检查模型初始化、cache 布局或环境 | “换 layerwise proxy 就能修复这个初始化 IndexError” |
| P/D 已 ready，发请求后失败 | 优先核对 proxy 协议、metadata、block 映射和传输 | “P 进程从来没起来” |
| 只看到 worker 退出、HCCL 超时、EngineCore 初始化失败 | 可能是连带报错，需要较早的具体异常链 | “HCCL 或 Mooncake 版本就是根因” |
| 没有 API ready 行或日志被截断 | 当前证据不足 | “没有 ready，所以肯定启动失败” |

采集器按每个文件的出现顺序保留样本，不保证多文件、多主机的全局时间顺序。必要时由操作者核对本地时钟和当前尝试边界，不能要求回传完整日志来排序。

## 2. 对已有怀疑逐项判定

### 单 tensor 注册越界

审查基点的 `mooncake_layerwise_connector.py:1264` 在 `create_kv_buffer()` 内取 `first_kv_cache_tuple[1]`。选择缓冲区的分支在约 1327 行；P D2RH worker 继承此注册路径，D 使用另一套 worker。

只有本次栈指向该函数，且被选中 cache 确实只有一个 tensor，才判定命中该问题。FA/C8 flags、`pd_head_ratio` 和注册顺序决定是否触发。不要把任意 `IndexError` 都归到这里。

GLM-5.2 W8A8 不等于 FA/KV C8；`enable_sparse_li_c8=true` 也不等于 `enable_fa_quant=true`。若 runtime flags 未观察到，使用 checkpoint 元数据只能形成假设。修复方向是按真实 cache 类型选缓冲区并处理不支持布局，不是简单捕获 IndexError 或把所有 C8 都关掉。

### MTP 层数

审查基点注册 MTP 层后没有同步扩充 `total_layers` 和事件数组。与计入 MTP 的 layerwise AscendStore 组合可触发 `Layerwise slot-release layout mismatch`；发送回调也可能提前返回。

本次 MTP 如果确实关闭，这不是本次错误的解释。不要把旧实验 MTP 失败当作当前运行事实。实际支持 MTP 的修复还需核对 P/D 层映射，不能只把两个数字调大。

### GLM-5.2 shared Indexer

`indexer_types` 的 shared 层可以没有独立 Indexer cache；检查该层的 `has_indexer` / `skip_topk`。只比较应该拥有独立 Indexer 的层。`indexer_types` 未采到时不能默认每层都有或每层都没有。

### overlap、MC2 和 Host pool

- 固定用户基线 `multistream_overlap_shared_expert=false`。若 runtime/argv 与基线冲突，报告冲突，不自动覆盖。
- `enable_fused_mc2`、`enable_prefill_mc2`、`enable_mc2_hierarchy_comm` 是不同设置；最终 MoE 分支还与设备、EP、token 数有关。只见 flag 不能证明每轮实际执行 MC2。
- `enable_fused_mc2=1` 在该分支会强制关闭 shared-expert overlap。不能把配置耦合后的实验称作完全独立的 A/B。
- D 的 `kv_offload_decode_config.use_fused_overlap=true` 是 shared Host pool 路径的要求，与 shared-expert multistream 开关不同。不要为了统一“关闭 overlap”把它关掉。
- 0.3.13 提供 shared_segment 不等于所有启动问题已修复。发行包、实际 Python 环境、CANN/HiXL 支持和 Python 代码异常需分别判断。

## 3. 核实 layerwise proxy

当前路径应使用实验分支内的 `examples/disaggregated_prefill_v1/load_balance_proxy_layerwise_server_example.py`。该文件头示例残留普通 proxy 名称，不能照头部命令启动，也不能只看进程显示名。

`facts.proxy` 记录从 PID argv 或启动脚本发现的实际入口、文件内容 SHA256、layerwise 路由标记是否存在。缺失、改名或未知入口都应标为待核实，不能仅因名称不匹配就宣布协议错误。

预期链路：

```text
Client → layerwise proxy → D 分配 Indexer/Main 目标块
       → D 回调 proxy /v1/metaserver
       → proxy 携带 D metadata 派发 P
       → P 计算并经 Mooncake 写入 D
       → 完成信号满足 D 的等待条件 → D 生成并返回
```

普通 proxy 是先 P 后 D，不能替代这条 D 先分配目标块的协议。使用错误 proxy 的旧请求结果不能作为 layerwise 传输正确性结论。

该分支 `/v1/metaserver` handler 会等待派发给 P 的任务；HTTP 200 的访问日志可能晚于 P 的实际执行。不要把这条访问日志的时间当作回调开始时间。`d_blocks_advertised` / `proxy_prefill_dispatch` / `p_blocks_mapped` 更适合补充定位，缺失时保持未知。

`remote_block_ids` 在线路上的顺序是 Indexer、Main，不一定等于 P 本地 group 顺序。回传 group 数、block 数量和 block size；本地比较映射后只回传是否一致，不能贴完整 ID 数组。只看 GET_META 成功或 DONE 发出，不能证明 D 已收到全部数据或输出正确。

P TP0 负责 Main D2RH。Indexer 的实际发送者要看 `is_indexer_sender` 分支；本分支注释/日志存在“所有 TP 发送”的旧表述，不能据此要求每个 TP 都有非空 payload。仍需收集非 TP0 worker 的状态和完成信号，以排查等待或 rank 选择问题。

## 4. 已获准重新实验时才执行

第一轮默认只读已有数据。得到操作者对修正 proxy/重新发请求的明确授权后：

1. 保存前一轮脱敏包。只修正 proxy 入口和对应路由设置；保持模型、MTP、MC2、量化、TP/DCP 等配置不变。不自行 kill 或重启其他人的进程。
2. 操作者在实验网内确认 P/D 各自健康，再确认 proxy 健康。健康检查响应只记录 HTTP 状态，不回显 body。脚本未自动进行这些检查，不能填成已完成。
3. 使用一个新建的公开、无敏感内容的短 prompt，例如“请回答 1 加 1 等于几。”，并发 1，最多生成 16 tokens，首轮非流式。通过 proxy 发送，不直接 P→D；完整响应仅在实验机本地保存。
4. 只回传 HTTP 状态、耗时、成功/失败、token 数（如有）和关联事件，不回传生成文本。本地检查输出是否符合预期后只报告判断。设置明确超时（建议单请求 120 秒），不自动无限重试，不追加 benchmark。
5. 失败即采集该轮三个角色包，标明最后观察到的阶段。首轮成功只说明这个短请求跑通，不说明长上下文、chunked prefill、并发、DCP 或性能已验证。

## 5. 信息不足时如何补采

只申请一个能够区分候选原因的观测点，例如 `create_kv_buffer` 调用前的 tuple 长度，或同一匿名请求的 D 完成状态。先检查过滤和覆盖预算是否造成缺项，必要时新增允许的结构化字段和合成测试。

禁止为拿到更多信息而把完整异常消息、上下文 100 行、全量启动参数或 DEBUG 请求体贴出机房。代码修改、运行时诊断点、重启和额外请求要列入下一步并由操作者批准；不得把本 skill 当作这些动作的通用授权。
