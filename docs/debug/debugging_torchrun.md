# 使用 debugpy + VS Code 调试 torchrun 分布式训练

本文介绍如何在 VS Code 中对 `torchrun` 启动的分布式训练脚本(以 SpecForge 的 `scripts/train_dflash.py` 为例)打断点调试,并解释 debugpy 与 VS Code 联调的底层原理。

---

## 1. 为什么 torchrun 不能像普通脚本一样直接调试

普通 Python 脚本可以在 VS Code 里直接按 `F5`(`request: launch`)启动调试。但 `torchrun` 有两个特殊之处,导致直接 launch 往往不好用:

1. **它是个启动器(launcher),不是训练进程本身。**
   `torchrun`(等价于 `python -m torch.distributed.run`)会读取 `--nproc_per_node` 等参数,然后 **fork/spawn 出 N 个子进程**,每个子进程才是真正运行 `train_dflash.py` 的 worker。你真正想打断点的代码运行在子进程里,而不是启动器进程里。

2. **多进程抢占。**
   N 个 worker 并行运行,如果每个都尝试监听同一个调试端口会冲突;而且大部分时候你只关心其中一个 rank 的执行流。

因此,调试 torchrun 的核心思路是:**让目标 worker 进程自己开一个调试服务端口,VS Code 以 `attach` 的方式连上去**,而不是让 VS Code 去 launch torchrun。

---

## 2. debugpy + VS Code 的联调原理

### 2.1 组件与角色

debugpy 是微软官方对 [Debug Adapter Protocol(DAP)](https://microsoft.github.io/debug-adapter-protocol/) 的 Python 实现。整条调试链路里有三类角色:

```
┌─────────────┐      DAP over TCP      ┌──────────────┐   in-process   ┌───────────────┐
│   VS Code   │  <------------------>  │ debugpy      │  injection ->  │  你的 Python   │
│ (DAP Client)│      (JSON 消息)        │ adapter      │                │  进程 (被调试)  │
└─────────────┘                        └──────────────┘                └───────────────┘
```

- **DAP Client**:VS Code(准确说是 Python Debugger 扩展)。它发出「设置断点 / 单步 / 求值变量」等请求。
- **debug adapter**:debugpy 启动的一个独立进程,负责在 DAP 协议消息和被调试进程之间转发。它常驻,使得即使被调试进程暂停,调试会话也能被管理(比如可以从 VS Code 侧终止进程)。
- **被调试进程(debuggee)**:你的训练进程。调用 `debugpy.listen()` 后,debugpy 会把调试后端「注入」到这个进程中(基于 pydevd),Hook 住字节码执行,从而实现断点暂停、栈帧读取、变量求值等。

三者通过 TCP 端口以 JSON 格式的 DAP 消息通信。

### 2.2 listen 与 connect 的区别(谁持有端口)

这是初学者最容易混淆的点。两种模式都对应 VS Code 里 `"request": "attach"`:

| 模式 | 谁打开/持有端口 | 谁先启动 | VS Code 配置 |
| --- | --- | --- | --- |
| **listen(推荐)** | 被调试进程调用 `debugpy.listen((host, port))` 打开端口并等待 | 先启动训练进程,再 attach | `attach` + `connect` |
| **connect** | VS Code(IDE)先监听端口 | 先启动 IDE 的监听,训练进程再 `debugpy.connect()` 连回来 | `attach` + `listen` |

> 命名很反直觉:代码里用 `listen`,VS Code 里就填 `connect`(去连接那个正在 listen 的端口);反之亦然。
> 一般推荐 **代码 listen / VS Code connect** 的组合——不需要先开着 IDE,训练进程随时可跑起来等你连。

### 2.3 `wait_for_client()` 的作用

```python
debugpy.listen(("0.0.0.0", 5678))
debugpy.wait_for_client()   # 阻塞,直到 VS Code attach 上来
```

`wait_for_client()` 会阻塞程序执行,直到调试器连接成功。这保证了你能在 **训练的第一行代码之前** 就断下来,不会错过初始化阶段的断点。如果不调用它,程序会立即全速运行,你可能来不及 attach。

### 2.4 远程 / SSH 场景

- `listen` 的 host 用 `0.0.0.0` 表示监听所有网卡,允许别的机器连入(**有安全风险**:任何能连上该端口的人都能在进程内执行任意代码,只应在可信网络里这么用)。
- 跨机器 / 跨容器时,推荐用 SSH 端口转发把远端端口映射到本地,然后 VS Code 连 `127.0.0.1`:

```bash
ssh -N -L 5678:127.0.0.1:5678 user@remote-host
```

- `pathMappings` 用于把本地代码路径映射到远端路径,保证断点能对应到正确的源码文件(本仓库里本地/远端路径一致,直接 `${workspaceFolder}` -> `${workspaceFolder}` 即可)。

---

## 3. 调试 torchrun 的三种方式

### 方式 A:代码内 listen + VS Code attach(本仓库采用,最推荐)

在训练脚本里插入一段「按需触发、只在目标 rank 生效」的调试钩子。SpecForge 的 `scripts/train_dflash.py` 已内置如下逻辑(位于 `main()` 开头):

```python
# 通过环境变量 SPECFORGE_DEBUG=1 打开;只有目标 rank 会等待调试器,
# 避免多个 torchrun worker 争抢同一个端口。
if os.environ.get("SPECFORGE_DEBUG") == "1":
    debug_rank = int(os.environ.get("SPECFORGE_DEBUG_RANK", "0"))
    if int(os.environ.get("LOCAL_RANK", "0")) == debug_rank:
        import debugpy

        port = int(os.environ.get("SPECFORGE_DEBUG_PORT", "5678"))
        debugpy.listen(("0.0.0.0", port))
        print(f"[debugpy] rank {debug_rank} waiting for VS Code attach on port {port} ...", flush=True)
        debugpy.wait_for_client()
```

`.vscode/launch.json`(本仓库已创建):

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Attach: train_dflash (rank 0)",
            "type": "debugpy",
            "request": "attach",
            "connect": { "host": "127.0.0.1", "port": 5678 },
            "justMyCode": false,
            "pathMappings": [
                { "localRoot": "${workspaceFolder}", "remoteRoot": "${workspaceFolder}" }
            ]
        }
    ]
}
```

使用步骤:

```bash
# 1. 安装 debugpy 到训练所用环境(uv 环境示例)
VIRTUAL_ENV=/path/to/venv uv pip install debugpy

# 2. 打开调试开关启动训练(建议先用 1 张卡)
SPECFORGE_DEBUG=1 bash examples/run_qwen3_8b_dflash_online.sh 1
# 看到:[debugpy] rank 0 waiting for VS Code attach on port 5678 ...
```

3. 在 VS Code 调试面板选择 **"Attach: train_dflash (rank 0)"**,按 `F5`,看到 `debugger attached` 即连上。之后在 `train_dflash.py`、`specforge/**`,甚至 sglang 库源码里下的断点都会命中(因为 `justMyCode: false`)。

调节参数:
- 调试别的 rank:`SPECFORGE_DEBUG_RANK=3`
- 换端口:`SPECFORGE_DEBUG_PORT=5679`(同时改 `launch.json` 里的 `port`)

**优点**:对任意 GPU 数都适用;精确控制调试哪个 rank;能进入第三方库源码。这是分布式场景最稳的做法。

### 方式 B:命令行 `python -m debugpy` 包裹 torchrun

不改代码,直接用 debugpy 以模块方式启动 torchrun:

```bash
python -m debugpy --listen 5678 --wait-for-client \
    -m torch.distributed.run --standalone --nproc_per_node 1 --nnodes 1 \
    scripts/train_dflash.py <你的训练参数...>
```

VS Code 用同样的 `attach` + `connect` 配置连接即可。

> 注意:debugpy 默认会「跟随子进程(subprocess following)」,所以即使 torchrun 会 spawn worker 子进程,断点通常也能命中。但当 `--nproc_per_node > 1` 时多个子进程会比较混乱,该方式最适合 **单卡(nproc=1)** 调试。参见 debugpy issue [#1311](https://github.com/microsoft/debugpy/issues/1311)。

### 方式 C:VS Code 直接 launch torch.distributed.run

在 `launch.json` 里直接以 `launch` 方式跑启动器(适合单卡、参数固定的场景):

```json
{
    "name": "Launch: train_dflash (1 GPU)",
    "type": "debugpy",
    "request": "launch",
    "module": "torch.distributed.run",
    "console": "integratedTerminal",
    "cwd": "${workspaceFolder}",
    "justMyCode": false,
    "env": {
        "PYTHONPATH": ".:${workspaceFolder}/specforge",
        "CUDA_VISIBLE_DEVICES": "0"
    },
    "args": [
        "--standalone", "--nproc_per_node", "1",
        "scripts/train_dflash.py",
        "--target-model-path", "/user/houzhenggang/models/Qwen3-8B"
        // ... 其余训练参数
    ]
}
```

按 `F5` 即可启动并调试。同样依赖 debugpy 的子进程跟随能力。

---

## 4. 方式 D:`torch.distributed.breakpoint()` 终端快速调试

除了图形化的 debugpy,PyTorch 自带一个轻量的分布式断点辅助函数 `torch.distributed.breakpoint()`。它需要你**手动写进代码里**——本质上是分布式版的内置 `breakpoint()`,在想暂停的那一行插入调用即可。

### 4.1 基本用法

```python
import torch.distributed as dist

# ... 训练代码 ...
dist.breakpoint(rank=0)   # 在这里暂停,进入 pdb
# ... 后续代码 ...
```

### 4.2 工作原理与规则

它内部的逻辑等价于(简化版):

```python
def breakpoint(rank: int = 0):
    if dist.get_rank() == rank:
        pdb.set_trace()          # 只有这个 rank 进入交互式调试器
    dist.barrier()               # 其余所有 rank 在此等待,直到选中 rank continue
```

由此得出两条必须遵守的规则:

1. **必须在所有 rank 上都执行到这一行,且传入相同的 `rank` 参数。** 被选中的 rank(默认 0)进入 `pdb` 接管终端输入,其余 rank 阻塞在 `barrier` 上等待。
2. **不要用 `if rank == 0:` 把它包起来。** 如果只有一个 rank 调到这条 barrier,其它 rank 永远等不到,会直接死锁 / 触发 NCCL 超时。它已经内部处理了「只让一个 rank 交互、其余等待」,你无条件调用即可。

### 4.3 它默认走终端 pdb,不是 VS Code

这是与前面几种方式最大的区别:`torch.distributed.breakpoint()` 默认弹出的是命令行 `pdb`,会在你启动 `torchrun` 的终端里出现 `(Pdb)` 提示符,用 `n`(下一行)/ `s`(步入)/ `c`(继续)/ `p var`(打印变量)等命令调试,**不会**自动连到 VS Code 图形界面。

- 想要终端轻量调试、快速定位死锁 → 用它,零配置。pdb 命令速查见 [pdb 使用指南](./pdb_guide.md)。
- 想要 VS Code 图形化断点、悬停看变量 → 用方式 A(`debugpy.listen` + attach)。

### 4.4 与方式 A 的对比

| | `torch.distributed.breakpoint()` | 方式 A(`debugpy.listen`) |
| --- | --- | --- |
| 是否手动加代码 | 是 | 是(已内置在 `main()`) |
| 调试界面 | 终端 pdb | VS Code 图形界面 |
| 多 rank 同步 | 原生支持(内部 barrier) | 靠 rank 门控选一个 |
| 适用场景 | 快速定位、死锁排查 | 完整图形化调试、单步查看变量 |

> 提示:`torch.distributed.breakpoint()` 用的是终端多路复用,确保你在**前台**运行 `torchrun`(不要 `nohup`/后台重定向),否则拿不到 `(Pdb)` 交互输入。
>
> 已知问题:在它的 `(Pdb)` 提示符下按方向键不生效、只出现 `^[[A`(它把 `sys.stdin` 重开为 `/dev/stdin`,导致 readline 行编辑失效)。解决办法见 [pdb 使用指南 · 常见问题 6.1](./pdb_guide.md#61-在-pdb-里按方向键上下不生效只出现-a--b)。

---

## 5. 分布式调试的注意事项

1. **优先单卡调试。** 多卡时,只有目标 rank 会在 attach 前停住,其它 rank 会继续跑到集合通信(如 `init_process_group`、`all_reduce`)处等待。如果你在断点停留太久,可能触发 **NCCL 超时**。可以调大 `--dist-timeout`,或干脆用 1 张卡。

2. **DataLoader 的 `num_workers` 设为 0。** 多进程数据加载在调试模式下常常无法正常 spawn 子进程,把 `num_workers=0` 可避免卡住。(见 [Programmer Sought 文章](https://www.programmersought.com/article/23007008352/))

3. **多 rank 都要断点、或排查死锁时,用 `torch.distributed.breakpoint()`。** 详见上文 [方式 D](#4-方式-dtorchdistributedbreakpoint-终端快速调试)。(参考 [PyTorch 论坛](https://discuss.pytorch.org/t/how-to-debug-pytorch-distributed/193575))

4. **端口残留。** 如果上一次调试进程没退干净,端口可能被占用,换端口或清理进程:

```bash
kill -9 $(pgrep -f "debugpy" | xargs echo)
```

5. **安全。** `listen(("0.0.0.0", ...))` 会对外暴露一个可执行任意代码的端口,仅在可信网络使用;跨网络务必走 SSH 隧道。

---

## 6. 参考资料

- [microsoft/debugpy — An implementation of the Debug Adapter Protocol for Python](https://github.com/microsoft/debugpy)
- [debugpy Wiki — API Reference(`listen` / `connect` / `wait_for_client` / subprocess following)](https://github.com/microsoft/debugpy/wiki/API-Reference)
- [debugpy Discussion #1688 — Terminology(client/server、adapter、listen vs connect 的解释)](https://github.com/microsoft/debugpy/discussions/1688)
- [VS Code 官方文档 — Python debugging(Remote debugging with SSH、attach 配置)](https://code.visualstudio.com/docs/python/debugging)
- [debugpy Issue #1311 — Support debugging console scripts(torchrun 的 `python -m debugpy ... -m torch.distributed.run` 用法)](https://github.com/microsoft/debugpy/issues/1311)
- [debugpy Issue #1615 — 多进程 PyTorch 调试卡住问题](https://github.com/microsoft/debugpy/issues/1615)
- [StackOverflow — How to make VS Code launch.json for a Python module(`module: torch.distributed.run`)](https://stackoverflow.com/questions/67518928/how-to-make-vscode-launch-json-for-a-python-module)
- [PyTorch Forums — How to debug pytorch distributed?(`torch.distributed.breakpoint`)](https://discuss.pytorch.org/t/how-to-debug-pytorch-distributed/193575)
- [Debug Adapter Protocol 规范](https://microsoft.github.io/debug-adapter-protocol/)
