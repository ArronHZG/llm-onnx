# pdb 使用指南(Python 命令行调试器)

`pdb` 是 Python 标准库自带的交互式源码调试器,无需安装。它支持设置(条件)断点、单步执行、查看调用栈、在任意栈帧上执行任意 Python 代码,以及事后(post-mortem)调试。

在没有图形界面的服务器 / 容器 / `torchrun` 终端里,pdb 是最顺手的调试工具。本仓库的 `torch.distributed.breakpoint()`(见 [debugging_torchrun.md](./debugging_torchrun.md) 方式 D)默认进入的就是 pdb。

---

## 1. 如何进入 pdb

### 1.1 在代码里插入断点(最常用)

```python
# Python 3.7+ 内置,推荐
breakpoint()

# 等价的老写法
import pdb; pdb.set_trace()
```

程序运行到这一行就会停下,弹出 `(Pdb)` 提示符。

> `breakpoint()` 的好处:可用环境变量 `PYTHONBREAKPOINT` 控制它调用哪个调试器,而无需改代码:
> - `PYTHONBREAKPOINT=0` —— 禁用所有 `breakpoint()`(适合生产环境)
> - `PYTHONBREAKPOINT=ipdb.set_trace` —— 换成 ipdb(带语法高亮、补全)
> - 不设置 —— 默认走 `pdb.set_trace`

### 1.2 从命令行启动(不改代码)

```bash
python -m pdb your_script.py [args...]
```

会停在脚本第一行,然后你用 `b`(设断点)+ `c`(继续)来控制。

### 1.3 崩溃后事后调试(post-mortem)

程序抛出未捕获异常后,直接在出错现场检查变量:

```bash
# 脚本异常退出时自动进入 post-mortem
python -m pdb your_script.py

# 或先跑到崩溃再进入
python -m pdb -c continue your_script.py
```

代码内也可以:

```python
import pdb
try:
    risky()
except Exception:
    pdb.post_mortem()   # 或异常发生后在交互式解释器里 pdb.pm()
```

> post-mortem 模式下**只能查看**崩溃瞬间的栈和变量,不能从崩溃点继续执行。

---

## 2. 核心命令速查

命令大多可缩写为首字母(下表括号即完整名),例如 `n` = `next`。**直接回车会重复上一条命令**(常用于连续 `n`)。

### 2.1 单步执行 / 继续

| 命令 | 缩写 | 作用 |
| --- | --- | --- |
| `next` | `n` | 执行当前行,**不进入**被调用的函数(step over) |
| `step` | `s` | 执行当前行,**进入**被调用的函数(step into) |
| `return` | `r` | 一直执行,直到**当前函数返回** |
| `continue` | `c` | 继续执行,直到遇到下一个断点(或结束) |
| `until [lineno]` | `unt` | 继续执行,直到行号大于当前行(常用于跳出循环);带参数则跑到指定行 |
| `jump lineno` | `j` | **跳到**指定行执行(跳过中间代码,危险,慎用) |

> `next` vs `step` 记忆:`next` 是「留在本函数」(跨过函数调用),`step` 是「钻进去」。

### 2.2 查看代码 / 栈 / 变量

| 命令 | 缩写 | 作用 |
| --- | --- | --- |
| `list [first,last]` | `l` | 显示当前行附近的源码(再按一次显示后续) |
| `longlist` | `ll` | 显示当前函数的完整源码 |
| `where` | `w` | 打印当前调用栈(stack trace) |
| `up [count]` | `u` | 上移一层栈帧(去调用者) |
| `down [count]` | `d` | 下移一层栈帧 |
| `args` | `a` | 打印当前函数的参数 |
| `print expr` | `p` | 求值并打印表达式 |
| `pp expr` | — | 美化打印(pretty-print),适合 dict/list/tensor |
| `whatis expr` | — | 打印表达式的类型 |
| `!statement` | — | 在当前栈帧执行任意 Python 语句(如 `!x = 5`) |
| `interact` | — | 打开一个带当前作用域的交互式 Python 解释器 |

> 任何 pdb **不认识**的输入都会被当作 Python 语句执行,所以大多数时候直接敲 `x`、`len(t)`、`obj.shape` 就能看值(与命令重名的变量才需要用 `p`/`!`)。

### 2.3 断点管理

| 命令 | 缩写 | 作用 |
| --- | --- | --- |
| `break` | `b` | 不带参数:列出所有断点 |
| `break lineno` | `b` | 在当前文件指定行设断点 |
| `break file:lineno` | `b` | 在指定文件的某行设断点 |
| `break func` | `b` | 在某函数入口设断点 |
| `break lineno, condition` | `b` | **条件断点**:仅当条件为真时中断,如 `b 42, x > 500` |
| `tbreak ...` | — | 一次性断点(命中后自动删除) |
| `clear [bpnum]` | `cl` | 删除断点(不带参数删全部) |
| `disable bpnum` | — | 临时禁用断点 |
| `enable bpnum` | — | 重新启用断点 |
| `ignore bpnum count` | — | 让某断点在接下来 count 次命中时忽略 |
| `commands bpnum` | — | 给断点绑定一组自动执行的命令(以 `end` 结束) |

条件断点示例:

```
(Pdb) b dataloader.py:88, batch_idx == 3
(Pdb) c
```

### 2.4 其它

| 命令 | 缩写 | 作用 |
| --- | --- | --- |
| `help [cmd]` | `h` | 查看命令列表 / 某命令的帮助 |
| `quit` | `q` | 退出调试器并终止程序 |
| `restart [args]` | — | (命令行模式下)重启程序,保留断点 |

---

## 3. 典型工作流示例

```python
def compute(values):
    total = 0
    for i, v in enumerate(values):
        breakpoint()          # 停在这里
        total += transform(v)
    return total
```

在 `(Pdb)` 里:

```
(Pdb) l              # 看当前代码上下文
(Pdb) a              # 看函数参数 values
(Pdb) p v            # 看当前循环变量
(Pdb) s              # 步入 transform(v) 看内部
(Pdb) w              # 确认调用栈
(Pdb) u              # 回到调用者栈帧看它的局部变量
(Pdb) pp locals()    # 美化打印当前所有局部变量
(Pdb) b 5, i == 10   # 设个条件断点,只在第 11 次循环停
(Pdb) c              # 继续跑到下一个断点
(Pdb) q              # 结束
```

---

## 4. 实用技巧

- **回车重复上一命令**:连续单步时按 `n` 一次,后面直接回车即可。
- **看不清代码就 `ll`**:`longlist` 显示整段函数,比反复 `l` 方便。
- **改变量再继续**:用 `!var = newval` 或直接 `var = newval` 现场改状态,验证假设。
- **跳出深层循环**:`until`(不带参数)一直跑到比当前行号大的行,常比手动数断点快。
- **条件断点代替 print 大法**:`b file:line, cond` 只在你关心的那次迭代 / 那个样本停。
- **崩溃排查首选 post-mortem**:`python -m pdb -c continue script.py`,崩了直接停在出错帧,`w` 看栈、`p` 看变量。
- **想要更好用的界面**:`pip install ipdb`,配 `PYTHONBREAKPOINT=ipdb.set_trace`,得到高亮 + 自动补全;或直接改用 VS Code + debugpy(见 [debugging_torchrun.md](./debugging_torchrun.md))。

---

## 5. 在分布式 / torchrun 场景下的注意事项

- pdb 是**终端交互**的,必须让程序在**前台**运行(不要 `nohup` / 后台 / 输出重定向),否则拿不到 `(Pdb)` 输入。
- 多进程(torchrun N 个 worker)里直接 `breakpoint()` 会让多个进程同时抢终端,输出混乱。请改用 `torch.distributed.breakpoint(rank=0)`,它只让指定 rank 进入 pdb,其余 rank 在 barrier 等待。详见 [debugging_torchrun.md 方式 D](./debugging_torchrun.md)。
- 若在断点停留太久,可能触发 NCCL 集合通信超时,可调大 `--dist-timeout` 或用单卡调试。

---

## 6. 常见问题

### 6.1 在 `(Pdb)` 里按方向键(上下)不生效,只出现 `^[[A` / `^[[B`

**现象**

```
(Pdb) ^[[A^[[A^[[B
```

按上下方向键调不出历史命令,也无法用左右键移动光标,终端只回显原始转义序列。

**原因**

方向键的行编辑 / 历史功能由 **readline** 提供。Python 内置的 `input()`(pdb 靠它读命令)**只有在 `sys.stdin` / `sys.stdout` 是「真正的」终端(`sys.__stdin__`)时才会启用 GNU readline**。

而 `torch.distributed.breakpoint()` 使用的 `_DistributedPdb` 为了支持多进程,把 `sys.stdin` 重新打开成了 `/dev/stdin`:

```python
class _DistributedPdb(pdb.Pdb):
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            with open("/dev/stdin") as sys.stdin:   # ← readline 因此失效
                pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin
```

这个新文件对象不是原始 tty,readline 不再接管,于是方向键失效。**这不是环境缺 readline**(`import readline` 正常),而是 torch 为多进程调试做的取舍。

**解决办法(按推荐度)**

1. **单卡调试改用内置 `breakpoint()`**(最简单,方向键原生可用)。用 `--nproc_per_node 1` 运行,代码里写内置 `breakpoint()` 而非 `torch.distributed.breakpoint()`,此时 `sys.stdin` 就是真实终端,历史 / 光标移动全部正常。

2. **必须多卡终端调试时,用 `rlwrap` 包住启动命令**。`rlwrap` 会给不使用 readline 的程序套上自己的行编辑与历史(只有 rank0 读输入,通常可用):

```bash
sudo apt-get install -y rlwrap
rlwrap bash examples/run_qwen3_8b_dflash_online.sh 8
```

3. **用 ipdb 获得高亮 + 补全**(走单卡 `breakpoint()` 路径;`torch.distributed.breakpoint` 内部写死了 pdb,`PYTHONBREAKPOINT` 对它无效):

```bash
VIRTUAL_ENV=/path/to/venv uv pip install ipdb
export PYTHONBREAKPOINT=ipdb.set_trace
```

4. **改用 VS Code + debugpy**。图形界面没有终端行编辑问题,多卡也能精确调某个 rank,见 [debugging_torchrun.md 方式 A](./debugging_torchrun.md)。

| 你的情况 | 推荐 |
| --- | --- |
| 只调一个 rank | 方案 1:单卡 + 内置 `breakpoint()`,零依赖 |
| 必须多卡终端调试 | 方案 2:`rlwrap` 包住启动命令 |
| 想要高亮 / 补全的终端调试 | 方案 3:ipdb(配单卡 `breakpoint()`) |
| 想要图形化、悬停看变量 | 方案 4:VS Code + debugpy |

---

## 7. 参考资料

- [pdb — The Python Debugger(Python 官方文档)](https://docs.python.org/3/library/pdb.html)
- [Real Python — Python Debugging With Pdb](https://realpython.com/python-debugging-pdb/)
- [DigitalOcean — How To Use the Python Debugger](https://www.digitalocean.com/community/tutorials/how-to-use-the-python-debugger)
- [PyGuides — Debugging Python with pdb(条件断点、post-mortem)](https://pyguides.dev/tutorials/intermediate-python/debugging-with-pdb/)
- [Python Debugger 速查表(Kapeli/Dash)](https://kapeli.com/cheat_sheets/Python_Debugger.docset/Contents/Resources/Documents/index)
- [PEP 553 — Built-in breakpoint()](https://peps.python.org/pep-0553/)
