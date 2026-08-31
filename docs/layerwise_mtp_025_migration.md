# 0.25rc1 layerwise MTP 迁移检查

## 基线与交付边界

- 当前代码基线：`mte_fuse_0723_mooncake_test_0827_add_block`，提交
  `16169875126291ffcd6bab3734e747cfc6e7a4ec`。
- 0.23 已验证参考：`feat/dsa-d2rh-pd-validated-base-20260826`，提交
  `b9bac806d4e7ce7818a32a984b7619040b3ff2d0`；“已验证”来自实验环境反馈，
  不代表本次在本机重新跑过 NPU 测试。
- 本次实验目标仍是 **0.25rc1、NPU、ARM、Python 3.12、GLM-5.2 W8A8**。
  分支里的 Dockerfile 默认 `VLLM_TAG=v0.25.1`，不能据此认定实验环境的版本。
  不使用该 Dockerfile 默认值重建并悄悄升级环境；先固定实际 vLLM、Ascend 的提交和导入版本。
- 只补 layerwise 路径；不改 blockwise 的元数据、拉取协议、分块地址算法。
  修改了共用的 layerwise 基类及 AscendStore，所以仍须检查无 MTP 的原有路径。
- 本补丁是注册/传输生命周期修复，**还不是 MTP 全功能或性能验收结论**。

## 已定位差距

| 环节 | 0.23 参考和 0.25 当前差距 | 本次处理 |
| --- | --- | --- |
| P 侧物理层数 | 0.23 注册后会补入额外缓存层；0.25 仍用 target 层数，MTP 回调可被 `current_layer >= total_layers` 跳过 | 按注册缓存的物理层去重计数 |
| Main / Indexer 拆分 | 0.25 同一物理层可以有两个缓存名字，不能直接用元数据条目数当层数 | 两个名字保留在同一个层下，Main 排在前面 |
| 带 `mtp` 的名字 | 原循环只保留最后一个名字；名字包含 `mtp` 且缓存拆分时会丢组件。实际名字若为 `model.layers.N`，不触发这个分支，但仍有层数问题 | 解析实际物理层号，保留全部组件；支持已有解析器的短 `mtp.N` 名字 |
| 发送事件 | 事件在注册前按 target 层数创建，仅增加 `total_layers` 会造成事件索引越界 | 在线程启动前为新增层补齐 pending / finished 事件 |
| AscendStore 多组 | 原代码用 target 层数过滤组内层号，额外的 MTP Main / Indexer 都可能被过滤 | 先算物理层并集，再重建组映射、任务数组、复用计划；不是组数求和 |
| 完成与失败 | 原最后一个 target 层可提前发 DONE；异常可能留下 pending，失败回调还存在重复发送路径 | 最后一个已注册物理层的两条传输腿结束后才发终态；异常记为失败并释放事件 |

`num_speculative_tokens=3` 不是三个 MTP 物理层。同一个 MTP 层可以连续运行三次；
本次按注册的物理层数安排一次 PD 传输，保留原先跳过额外 drafting 回调的行为。
这只修复了“计数及排队”，尚不能证明后续 drafting 写回与异步传输之间不存在竞争。

`add_block` 相比先前 `0827` 分支的新增提交没有修改上述 layerwise 文件。
因此 blockwise 的 eager 冒烟成功不能直接覆盖这些 layerwise 问题。

## 修改范围

1. `vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py`：
   注册物理层、绑定 Main / Indexer 名字、补齐发送事件。
2. `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py`：
   注册前更新实际层数、组映射和复用计划；复用已有层号解析函数。
3. `vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_to_dram_connector.py`：
   收拢两条传输腿的完成/失败处理，保证异常后释放源缓存事件。
4. `tests/ut/kv_offload/test_layerwise_mtp_registration.py`：注册、排队、组映射、
   复用计划、末层终态和异常清理的回归测试。

不修改 ModelRunner、MTP 权重映射、W8A8 算子、scheduler、Host pool 分配器和环境变量。
Host pool / KV tensor 的容量与存储规划仍由 runner 完成；connector 绑定、注册并传输这些缓存。
更新 Python 事件数组不是重新分配模型 KV 内存。

0.25 已有 MTP compute / offload 适配：proposer 的 DeepSeek MTP 双返回值处理、
多步 drafting 的请求元数据更新、manager 的额外层注册、以及多 query 的 offload 检查。
不能把这些当作全部缺失而整体搬入 0.23 的实现。

## 尚需验证的风险

| 风险 | 为什么本次补丁不能直接覆盖 | 验证 / 后续修改边界 |
| --- | --- | --- |
| P / D MTP 缓存不一致 | P 未注册的 MTP KV 不会凭空传给 D；P / D 的物理层和组件必须匹配 | 首轮 P、D 都启用 MTP，比较物理层与组件摘要；不同 drafting 步数后测 |
| 连续 drafting 写源缓存 | 当前仍只发送每个物理层第一次回调；后续 drafting 可能与异步读源并发 | 先 P=1、D=1；再 P=1、D=3；最后 P=3、D=3。若有竞争，另补按实际层号的写入等待，不能只在 `save` 后等待 |
| AscendStore 复用等待 | MultiConnector 接入外部等待器后会绕过 provider 的 layer-entry 等待 | 后续若加 MTP 重入保护，必须覆盖 sink 路径，不能只修改 PD worker 的空 `wait_for_layer_load` |
| verify / reject / cache 命中 | D 侧多 query token-to-request、slot、LRU/selection 失效处理可能出现静默错读 | 多请求、部分接受/全部拒绝、前缀命中、跨块边界；发现问题再改 proposer / offload manager |
| 真正多个 MTP 物理层 | 本补丁注册可以保留多个物理层，但 manager 的 `mtp_layer_id` 仍只标记末层 | GLM 实际配置先确认物理 MTP 层数；多物理层模型不在首轮支持声明内 |
| DCP / PCP / PP | 本轮没有重新证明层号、token/block 映射和通信组在这些模式下正确 | 首轮 PP=1、DCP size=1、PCP size=1；DCP 扩展另做，不能把“关闭”写成 size=0 |
| graph / overlap / MC2 | Host/GVA 地址、事件顺序、动态 token 数与图捕获需要实际硬件验证 | eager 通过后单独恢复图模式和 overlap；MC2 保持当前实验基线 |
| 量化与内存 | W8A8 不等于 KV cache 量化；Indexer 和 Main 的 dtype、tuple 长度不同；MTP 增加 query buffer 压力 | 记录 dtype/shape/组件数及峰值内存摘要，不自行改权重量化或额外打开 KV/C8 量化 |

注册修复不包含短 `mtp.N` 名字在 D2RH 全链路的通用适配；D2RH 其他函数仍有
`layers.N` 解析要求。GLM 的实际注册名字必须核对，不能只凭注册单测宣称别的模型也支持。

## 实验机操作顺序

1. 固定当前可冒烟环境，不升级 Mooncake、vLLM、Ascend、torch-npu、CANN 或 transformers。
   保存实际包版本、代码提交、安装包与源码是否一致、模型配置的物理层数。
   这些信息先在机内核对，只外传版本号/提交号/布尔校验结果等脱敏摘要。
2. 使用 layerwise connector 和
   `examples/disaggregated_prefill_v1/load_balance_proxy_layerwise_server_example.py`。
   沿用已经正确的地址和启动参数；不要混用 blockwise proxy / connector。
   不启用 MTP 时移除 speculative 配置，不要用非法的 token 数模拟关闭。
3. 保持 eager、`multistream_overlap_shared_expert=false`；其他 overlap 开关按当前关闭基线，
   MC2 按当前开启基线。PP=1、DCP size=1、PCP size=1，TP 拓扑先保持不变。
4. 先跑下列 UT。失败时只提取测试名、异常类型、相关自有代码行和脱敏原因，
   不把整个 pytest 输出直接交给外部 agent。

```bash
pytest -q \
  tests/ut/kv_offload/test_layerwise_mtp_registration.py \
  tests/ut/kv_offload/test_mooncake_to_dram_asymmetric_push.py \
  tests/ut/kv_offload/test_mooncake_layerwise_connector.py \
  tests/ut/kv_offload/test_ascend_multi_connector.py \
  tests/ut/distributed/ascend_store/test_pool_worker.py
```

5. 依次测试：双方 MTP 关闭回归 → 双方 1 步 → P=1/D=3 → 双方 3 步。
   这里的数字均是 `num_speculative_tokens`，不是物理层数。任一阶段失败即停止扩大配置。
6. 每阶段先单请求，再多请求；覆盖短输入、跨缓存块边界、chunked prefill、前缀命中、
   长生成和重复请求。先用固定输入、seed、greedy 设置比较同一环境下 MTP 开/关结果，
   记录一致性及首个分歧位置统计，不上传输入、输出文本或 token 序列。
7. 除健康检查外必须有实际生成成功；核对 MTP Main / Indexer 都完成传输后才出现 DONE，
   故障注入时是 FAILED，且源事件最终释放。校验只比较机内数据，输出匹配/不匹配及计数。
8. 全部通过后再恢复 graph；DCP>1 和 overlap 各自单独增加，不一起打开定位不了原因。

仅允许外传：版本与提交摘要、使用的路径类型、开关、层数/组件数/dtype/shape、
测试成功失败数、错误类别、延迟/内存/接受率统计、脱敏分析。
**不上传完整日志、提示词、模型输出、token/block ID 序列、地址、IP、路径、请求标识或凭据。**
原始日志必须在机内过滤；不能先发给外部 agent 再让其删减。

## 本机验证状态

- 已通过 19 项隔离源码方法回归：执行上述测试文件的用例，加载实际待测方法，
  用假 tensor/engine/context 隔离 NPU、Mooncake 和 vLLM 的运行依赖。
- 该结果不等于完整模块导入测试，不覆盖真实内存、网络、进程、设备流、graph 或模型精度。
- Python 3.12 语法检查、新增测试的 Ruff 检查及 `git diff --check` 通过。
- 在隔离工作树执行了全仓 `bash format.sh ci`，未通过；随后在完全未修改的
  `161698751` 基线上重跑，复现相同类别的失败：Ruff、拼写、clang-format、
  Python package init、禁用 import 和 symbolic shape 检查。
  两次格式检查均涉及 42 个存量文件；没有把这些自动格式化改动带入本补丁。
  新增测试和本说明均未被全仓检查器修改。不能将本次交付标为“全仓 CI 通过”。
- 本机没有 NPU / torch-npu / 实验模型，完整 UT、服务冒烟、MTP 准确性、DCP 及性能测试待实验机完成。
