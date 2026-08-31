# 0.25rc1 blockwise MTP 迁移

## 与 layerwise 的关系

本分支直接基于 GitCode `mte_fuse_0723_mooncake_test_0827_add_block` 的
`d1bf0bad215f9fd552f7251cc952e12b9256af3a`。该提交在旧基线 `161698751` 上修复
Decode graph 的 external planner：使用紧凑的 CPU plan staging，分别处理 planner 与
operator membership 的偏移/stride，并校验 CPU、dtype、shape、连续性。
本补丁不修改该 manager 文件或其图模式测试，完整保留这次更新。

已用 `git merge-tree --write-tree` 检查最新基线与 `codex/layerwise-mtp-025`
的 `63218b48b`，没有 Git 文本冲突。但两种 PD 协议不能当作相同的运行路径：

| 项目 | layerwise | 本次 blockwise |
| --- | --- | --- |
| 传输 | P 按层 Push，最后一层发终态 | D 按请求 Pull，读完必要组件后报告接收完成 |
| MTP 完整性 | 层注册、逐层事件、末层 DONE | 缓存身份、源/目标地址、block 组号和覆盖范围 |
| Main Host pool | 绑定 runner 分配的池 | 同样绑定 runner 的池，由 D TP0 写 Main |

原 layerwise 分支不改、不合并。本分支没有 cherry-pick 其注册、AscendStore 或 DONE 修改。
源码可以共存；同一个 PD 请求不要同时挂两种 PD connector 去写同一个池。
这不禁止已有职责独立的 prefix-cache connector，但组合使用需另做生命周期验收。

## 本次修复

### 1. 用缓存身份匹配 P / D

原 `kv_group2layeridx` 的内部序号会受 group 顺序、重复层号和带 `mtp` 的名字影响。
这些序号不一定是物理层号，也不保证 P / D 相同。因此不能移植 layerwise 的计数补丁，
然后继续用相同数组下标读写。

在 `MooncakeAgentMetadata` 增加可选 `dsa_cache_layout`：

```text
main:64    -> (实际 wire row, 起始 tensor position, 2)
indexer:64 -> (实际 wire row, 起始 tensor position, 1 或 2)
```

物理层号从 `model.layers.N`、`model.layers.N.mtp_block` 或短 `mtp.N` 名字解析。
短名字的 N 加 target 层数；`num_speculative_tokens=3` 不参与物理层编号。
支持 Main/Indexer 分开注册，也支持 P 将 Indexer 放在 Main tuple 的第 3/4 项。
描述符记录真实追加位置，不重排现有地址数组、不改变原 group 编号。

D 将 P 地址映射到本地需要的组件，再按原路径做 Indexer D2D 和 Main D2RH。
在开始读取之前检查所需 Main 和 MTP Indexer 是否存在、组件数是否一致、四组地址数组是否完整。
GLM 共享 target Indexer 的缺省行为保留，但必须存在对应 Main；MTP Indexer 不走这个豁免。

### 2. 明确 block 组号

原 D 默认第一组是 Indexer、最后一组是 Main；P 的组顺序不保证如此。
P 现在根据组件名确定实际组号，在请求 `kv_transfer_params` 中发送
`dsa_block_group_ids={main: ..., indexer: ...}`，不再依赖 DEBUG 开关。
D 按该映射选取原 `remote_block_ids` 中的列表。

P 原有按 prompt 长度裁掉额外 MTP 预留 blocks 的行为保留。传输的是 prompt KV，
不是把所有用于 speculative lookahead 的容量都复制过去。

### 3. 不允许漏传后报成功

- 缺少所需 MTP、缺少 Main、key/scale 组件数量不匹配：明确报错。
- Indexer 目标容量不能覆盖全部源 pages：报错，不再使用 `min(...)` 静默截断。
- 一个目标行只有部分有效 page 是允许的；拒绝的是实际源 pages 超过目标容量。
- `RECEIVE_COMPLETE` 仍在两条所需传输腿完成后产生；非 owner 只拉自己的 Indexer。
- `DONE_RECVING_MSG` 是释放 P 源 blocks 的通知，失败清理也会发送，不能当作成功证据。
  TE 返回负值仍走已有 `TRANSFER_FAILED` / 重算流程；布局校验异常走已有错误队列，明确中止，
  不伪装成接收成功。

## 兼容性与未覆盖范围

- 新字段是可选握手扩展，原地址数组保持不变。真实 msgspec 测试覆盖新读旧、旧读新编解码。
  无 MTP 旧对端保留原 positional 解析入口；这不等于所有跨版本拓扑已经实测。
- **MTP 请 P、D 同时升级到本分支。** 旧 P 没有描述符时，新 D 的 MTP 接收会明确拒绝。
  P 未启用/未注册 MTP 而 D 要求 MTP，也会报缺少缓存。proxy 必须原样转发新增请求字段。
- 本轮仍是一组 Main、一组 Indexer，或二者同属一个 manager group。
  如果 MTP 因不同 KV 规格被放进额外 manager group，当前每请求两套 block-ID 列表不足以描述它。
  现在明确拒绝这种配置；下一阶段需要扩展命令为逐层/逐组 block-ID 路由，不能只取消校验。
- GLM-5.2 的实际 MTP 层数、Indexer dtype/scale、KV 分组必须从目标环境核对。
  W8A8 是权重量化信息，不能据此断定 KV 或 Indexer 是否 C8。
- 本轮不扩展 DCP。上游 blockwise D 侧仍要求 `DCP * PCP == 1`、Decode PP=1。
  首轮 P / D 都使用 PP=1，DCP / PCP 的关闭值为 size=1。
- MTP 真正多物理层、更多分组、不同 P/D 模型缓存规格不在本轮支持声明内。
- P 源缓存发布与 drafting 完成的先后、AsyncScheduler、拒绝回退、prefix-cache 命中、
  graph replay 的 planner 数据更新，都需要实际 NPU 验证。没有用新增全局同步掩盖这些风险。

## 修改范围

- `vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py`：握手扩展、
  显式组号、接收前映射与完整性检查。
- `vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_dsa_layout.py`：独立的缓存身份与映射逻辑。
- `tests/ut/kv_offload/test_mooncake_dsa_mtp_layout.py`：新增回归；
  `test_mooncake_dsa_shared_pool.py`：补齐现有测试夹具的新元数据。

没有修改 layerwise、AscendStore、ModelRunner、Host pool 分配、MTP 权重加载、量化算子或环境变量。
映射发生在每次 PD 接收的 CPU 控制路径，不在每个 Decode token 的设备计算路径中。

## 实验机验证顺序

1. 固定 **0.25rc1、ARM、Python 3.12、NPU、GLM-5.2 W8A8** 环境；
   不根据 Dockerfile 的 `v0.25.1` 默认值重建并混入另一个版本。
   本地核对 vLLM、Ascend、Mooncake、torch-npu、CANN 的版本/提交及实际导入来源。
2. 先提取 target / MTP 物理层数、按 Main/Indexer 分类的 manager group 数量、
   每种组件的 dtype/shape/tuple 长度。若同一组件有多个 manager group，停止，报告该布局边界。
3. P / D 使用同一提交，选择原 blockwise `MooncakeConnectorV1` 路径和 `dsa_pd_offload=true`。
   使用 `examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py` 这一类普通 PD proxy，
   不沿用 layerwise proxy；确认其转发完整 `kv_transfer_params`。
4. 保留已有 Mooncake backend、runner Host pool、TP 拓扑、MC2 配置；
   `multistream_overlap_shared_expert=false`，其他 overlap 暂按现有关闭基线。
5. 先执行 UT：

```bash
pytest -q \
  tests/ut/kv_offload/test_mooncake_dsa_mtp_layout.py \
  tests/ut/kv_offload/test_mooncake_dsa_shared_pool.py \
  tests/ut/kv_offload/test_mooncake_dsa_metadata.py \
  tests/ut/kv_offload/test_mooncake_connector.py \
  tests/ut/kv_offload/test_kv_offload_decode_external_plan.py
```

6. 无 MTP eager 回归 → 无 MTP graph 回归（确认保留上游图修复）→
   P/D 都 MTP 1 步 eager → P=1/D=3 步 eager → 双方 3 步 eager → 对应 graph。
   数字是 speculative tokens 数，不是 TP 或物理层数。任一步失败，不继续堆叠开关。
7. 每阶段覆盖单请求/多请求、跨 block 边界、chunked prefill、前缀命中、长生成、
   接受与拒绝回退。用同一版本固定输入、seed、greedy 比较 MTP 开/关，统计首个分歧位置、
   错误数、接受率、延迟和峰值内存。服务启动或 graph capture 成功不等于生成正确。
8. 核对每个需要的 MTP Main / Indexer 都在传输范围内；owner 与非 owner 都收到正确终态。
   补做缺少 MTP 描述符、组件数量不匹配和传输失败注入，确认不会产生 `RECEIVE_COMPLETE`。

日志仅在机内处理。允许外传版本/提交、开关、层数、组数、dtype/shape、测试计数及脱敏分析。
不外传完整日志、prompt、输出文本、token/block IDs、地址、IP、路径、请求标识或凭据。
不能把原始日志交给外部 agent 后再让其过滤。

## 本机验证记录

- 33 项隔离源码回归通过：使用真实映射模块、提取的实际 scheduler / receiver 方法和真实 msgspec。
- 四个 Python 文件的 Python 3.12 语法检查、修改的 connector 与新增文件 Ruff 检查通过。
- 全仓 `bash format.sh ci` 已执行但未通过；在未修改的 `d1bf0bad2` 上重跑，
  复现相同类别的 Ruff、拼写、clang-format、package init、禁用 import 和 symbolic shape 问题。
  没有带入自动格式化产生的存量改动；新增映射模块、测试和本文档未被检查器修改。
- 这不是完整 vLLM 模块导入测试，也不是 NPU、RDMA、graph 或真实权重测试。
- 完整 UT、MTP 生成准确性、图模式、性能测试待实验机执行；不声明 MTP 已实机验证通过。
