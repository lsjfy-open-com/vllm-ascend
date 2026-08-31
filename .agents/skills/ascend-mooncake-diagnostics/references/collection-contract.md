# 本地 manifest 与允许回传字段

manifest 只在实验机保存。下面的 `/actual/...` 均须替换，不可照抄执行。不要把 manifest 当作分析附件。

`facts.declared_test_baseline` 固定记录用户确认的 **0.25rc1**，来源是用户声明。`environment.packages` 独立记录实际采集环境的包版本，保留 `rc` 后缀；两个 repository 字段记录源码 commit。三者不能互相替代，也不能以 0.23 测试结果补缺项。

```json
{
  "schema_version": 1,
  "role": "P",
  "instance": 0,
  "ascend_repo": "/actual/experiment/vllm-ascend",
  "vllm_repo": "/actual/experiment/vllm",
  "correlation_key_file": "/actual/private-diagnostics/correlation.key",
  "logs": ["/actual/current-attempt/worker-tp0.log", "/actual/current-attempt/worker-tp1.log"],
  "pid": null,
  "command_file": "/actual/launch-p.sh",
  "command_cwd": "/actual/launch-working-directory",
  "model_config_file": "/actual/glm52/config.json",
  "quant_config_file": "/actual/glm52/quant_model_description.json",
  "version_files": {
    "cann": "/actual/cann/version.info",
    "hixl": "/actual/hixl/version.info"
  },
  "max_bytes_per_log": 134217728
}
```

## 输入规则

| 字段 | 怎么填写 |
| --- | --- |
| `role` / `instance` | 只允许 `P`、`D`、`PROXY` 和 0–999 实例号；不能填主机名。多个 TP worker 属于同一服务实例。 |
| `ascend_repo` | 实际实验 checkout，不是诊断工具目录。收集 commit、dirty 状态和几个已知源码文件的摘要，不导出 diff 或路径。 |
| `vllm_repo` | 有源码 checkout 才填写；只有安装包时填 `null`。 |
| `logs` | 同一次尝试的当前角色日志路径。不能把昨天和今天、P 和 D 混成一包；多 rank 混写日志可以解析带 rank 前缀的异常。 |
| `pid` | 能核实的当前服务进程 PID；优先于 `command_file`。读取 `/proc/PID/cmdline` 后只保留允许的参数。PID 退出/无权限时报固定错误，不回退到猜测的进程。 |
| `command_file` | 原有启动脚本路径；只用 shlex 解析，不执行，也不展开 shell 变量。未解析到参数不表示参数关闭。 |
| `command_cwd` | 启动脚本中的相对 proxy 路径所对应的工作目录。当前 PID 可用时使用 `/proc/PID/cwd`。 |
| `model_config_file` / `quant_config_file` | 本地 checkpoint 的小型 JSON 元数据。只取模型类别、维度、owned/shared 计数和量化标志/权重量化计数。绝不读取权重。 |
| `version_files` | 已知 CANN、HiXL、driver 的本地版本文件，只解析 `Version=`、`version=`、`version_dir=` 的数字版本。没有文件或格式未知就不填/标未知，不导出完整安装信息。 |
| `correlation_key_file` | 同一实验所有角色共享的随机 32 字节 key，以 64 位十六进制保存。不要用凭证、API key 或机器标识代替。 |

大于 16 MiB 的 JSON/脚本输入会拒绝；这时不能改为直接打印文件。让操作者确认是否选错了文件。路径不存在或解析失败仅输出固定错误码，不把原异常文本带出机器。

## 三类配置不能混淆

1. **启动参数**：`command.values`。只保留 TP/DP/PP/PCP/DCP、block size、graph、MTP token 数、MC2、量化和 connector 结构等允许字段；connector 顺序保留。模型路径、地址、API key、engine ID、未知额外配置不导出。重复参数名在 `repeated_flags` 中列出，不能把多个启动命令混在一份脚本中当成有效配置。
2. **Checkpoint 元数据推导**：`fa_quant_type` 非空决定 `enable_fa_quant`，`indexer_quant_type` 非空决定 `enable_indexer_quant`，`kv_cache_type == C8` 决定 `enable_c8_quant`。这是被审查分支的逻辑，不是 W8A8 名称推导。实际文件缺失时不产生这些结论。
3. **运行时观测**：只处理已有日志里的 `ASCEND_DIAG_FACTS=` JSON 标记。没有这个标记是正常情况，本轮不自动加日志、不修改源码。先把异常、配置和 proxy 证据回传，待下一轮决定是否需要下面的最小观测。

## 需要下一轮运行时观测时

得到一次性诊断授权后，观察点应在对应函数真正持有这些对象时，不能额外加载一套模型“模拟有效配置”。

- 若栈在 `create_kv_buffer`：在调用前观察被选中层的 group/spec 类型、tuple 长度、tensor shape/dtype，以及该 worker 的 `enable_kv_quant`、`enable_c8_quant`、`pd_head_ratio`。`enable_kv_quant` 来源是 quant config 的 `enable_fa_quant`，对外统一记后者。
- 若栈在 cache compose：观察 main tuple 与真实 Indexer tuple；至少取一个 owned Indexer 层、一个 shared 层。没有 Indexer 的 shared 层不强行访问 `self.indexer.k_cache`。
- 若层数不一致：观察主模型层数、实际 MTP 层数、`total_layers`、完成事件数组长度及 AscendStore 层数。
- 只取元数据，不调用 `.item()`、`.cpu()`、`.tolist()`、checksum 或打印 tensor 内容；只在初始化点记录一次。不要在每 token 路径加日志。

允许的结构示例（这些是示例值，**不能作为本次证据**）：

```json
{
  "flags": {
    "enable_fa_quant": false,
    "enable_c8_quant": false,
    "enable_sparse_sfa_c8": false,
    "enable_sparse_li_c8": true,
    "multistream_overlap_shared_expert": false
  },
  "total_layers": 1,
  "event_count": 1,
  "layers": [
    {
      "layer_index": 0,
      "group_id": 0,
      "kind": "indexer",
      "tuple_len": 2,
      "shapes": [[2, 128, 1, 128], [2, 128, 1, 1]],
      "dtypes": ["int8", "float16"],
      "block_size": 128,
      "has_indexer": true,
      "skip_topk": false
    }
  ]
}
```

该 JSON 标记仍经采集器允许字段过滤，不能混入路径、ptr、weights 或请求内容。已有普通 `layer: ... num_blocks: ... block_shape: ...` 日志只说明观察到了一个注册 tensor 的 shape，不证明 tuple 长度或 dtype。

## 覆盖范围与保密限制

- 每包最多保留 12 类异常，每条异常链最多 6 段、每段最多 48 个栈帧；超长栈保留入口和末尾，明确标记 `incomplete`。重复异常保留计数和最多 64 个 rank 组合。
- 事件最多 96 条，保留开始和末尾样本；总出现次数另外统计。cache 最多 24 个样本。首次因果证据优先，不导出整段日志。
- 地址、消息原文、源码行、block ID 数组不导出；仅记录 block 数量、shape、允许的数字参数。请求 ID 使用同一实验 key 做 HMAC，不导出原 ID。
- 栈仅保留已知文件名和函数名，其他内容显示 `other_file` / `other_function`。line 是原输入文件行号，source 是 manifest 中日志的从零开始序号，真实文件路径留在机房。
- `clock` 是日志原有的数字时间，没有日期/时区则不补猜。跨主机顺序默认未校验，不按不同机器的时钟直接确定全局根因。
- 过滤器会损失一部分定位信息，这是主动保密边界。需要新的错误类别/函数名时维护允许列表并重新测试，而不是放行原始文本。
- 原始日志本身是不可信证据：它可能被截断、包含伪造的标记或请求内引用的异常文本。格式匹配只表明观察到记录，不能取代运行状态核验。
