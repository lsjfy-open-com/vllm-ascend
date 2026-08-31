# 可直接交给实验机 Cursor 的任务

把下面代码块中的文字作为 Cursor 任务。先取得 skill 目录，再执行。不要把下面的“取得目录”理解为允许切换正在测试的服务分支。

```text
请先完整阅读本地 ascend-mooncake-diagnostics/SKILL.md，并按其中顺序执行第一轮只读采集。

目标是排查 mte_fuse_0723_mooncake_test_0827 分支上的 Mooncake layerwise PD：
本轮全部测试基于 0.25rc1。分别核实 vLLM / vLLM Ascend 的安装版本和 commit，禁止混入 0.23 的结果。
GLM-5.2 W8A8，multistream_overlap_shared_expert 已关闭；此前接近成功的配置关闭 MTP、开启 MC2。
昨天用错了普通 proxy，需要区分进程初始化失败和首请求协议失败。

只允许本地 Python 脚本读取原始日志、启动脚本、模型 JSON 和进程参数。
这些内容不得通过工具输出、IDE 打开或直接文件读取进入你的上下文。
你只能读取 collect.py 生成的 facts.json、evidence.txt、analysis.md。

先确认当前尝试的 P/D/proxy 日志路径、实验仓库路径和服务 Python 路径。
不知道就只问路径，不让我粘贴原始内容。每个角色分别采集，同一尝试在实验网内使用同一个 correlation key。
不要在另一个 venv 采版本，不要把实验 checkout 切到诊断分支。

本轮不升级、不安装依赖、不改服务代码、不重启、不发请求、不开放 DEBUG、不上传数据。
采集后按分析指南和模板写出事实、假设、缺失证据、最小下一步，每项结论引用带角色/实例的证据编号。
W8A8 不能当成 KV C8；shared Indexer 层没有独立 tensor 不能直接判错；没看到日志不能当成没发生。
最终仅准备三个允许回传文件；manifest、key、raw logs、模型文件必须留在实验机。
若需修正 proxy 后重测，先给出最小计划，等我批准该轮操作。
```

## 取得 skill 目录，不改实验 checkout

允许访问用户仓库时，在已有实验仓库内执行以下取文件操作。命令只更新 FETCH_HEAD 并导出工具目录，不执行 checkout/reset，不改变工作区代码：

```bash
set -o pipefail
DIAG_TOOLS=$(mktemp -d /tmp/ascend-diag-tools.XXXXXX)
git fetch https://github.com/lsjfy-open-com/vllm-ascend.git codex/mooncake-diagnostics-skill
git archive FETCH_HEAD .agents/skills/ascend-mooncake-diagnostics | tar -x -C "$DIAG_TOOLS"
DIAG_SKILL="$DIAG_TOOLS/.agents/skills/ascend-mooncake-diagnostics"
```

显式让 Cursor 阅读 `$DIAG_SKILL/SKILL.md`；不依赖 Cursor 自动发现 skill 的目录规则。若实验仓库有未完成的 Git 操作，不在其中 fetch，改由操作者在独立临时仓库取文件。

不能联网时，由操作者把该 skill 目录通过允许的离线方式拷入实验网即可。采集脚本运行不需要 pip、模型下载、GitHub API 或任何联网资源；Git 仅用于读取本地 checkout 的版本信息。

这份任务只授权采集。不要使用 `git push` 把任何实验数据回传到该仓库。
