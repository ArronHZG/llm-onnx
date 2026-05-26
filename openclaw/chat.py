"""
本地大模型对话程序 - 连接远程 SGLang 服务 (Qwen3.6-27B)

功能：
  - 交互式多轮对话（自动维护上下文）
  - 支持流式输出（打字机效果）
  - 支持 reasoning 模式
  - 命令：/clear /quit /mode /reason /tokens /save /load /help

使用方式:
    python3 chat.py
"""

import json
import requests
import sys
import os
import datetime

# ============================================================
# 配置
# ============================================================

SGLANG_BASE_URL = "http://33.29.18.248:44400/v1"
MODEL_ID = "/home/hadoop-djst-algoplat/model/Qwen/Qwen3.5-35B-A3B"
MAX_HISTORY_TURNS = 50
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chats")

class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    RED     = "\033[31m"
    GREEN   = "\033[32m"
    YELLOW  = "\033[33m"
    MAGENTA = "\033[35m"
    CYAN    = "\033[36m"


# ============================================================
# 对话管理器
# ============================================================

class ChatSession:

    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.messages: list[dict] = []
        self.stream_mode = True
        self.show_reasoning = True
        self.max_tokens = 8192  # Qwen3.5 MoE 需要足够空间输出 reasoning + content
        self.temperature = 0.7
        self.top_p = 0.9
        self.token_count = {"prompt": 0, "completion": 0, "total": 0}

    def add_message(self, role: str, content: str):
        if content is None:
            content = ""
        self.messages.append({"role": role, "content": content})
        if len(self.messages) > MAX_HISTORY_TURNS * 2:
            if self.messages and self.messages[0]["role"] == "system":
                self.messages = [self.messages[0]] + self.messages[-(MAX_HISTORY_TURNS * 2) + 1:]
            else:
                self.messages = self.messages[-(MAX_HISTORY_TURNS * 2):]

    def clear(self):
        self.messages.clear()
        self.token_count = {"prompt": 0, "completion": 0, "total": 0}
        print(f"{C.GREEN}✅ 已清空对话上下文{C.RESET}")

    @property
    def turn_count(self) -> int:
        return len([m for m in self.messages if m["role"] == "user"])

    # ---- API 调用 ----

    def _build_payload(self, user_input: str) -> dict:
        msgs = self.messages + [{"role": "user", "content": user_input}]
        return {
            "model": self.model,
            "messages": msgs,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stream": self.stream_mode,
        }

    def _clean_text(self, text: str) -> str:
        """清理文本中的多余空白，提取可读内容"""
        if not text:
            return ""
        import re
        # 尝试提取连续的可见文本块
        text = re.sub(r'[^\u4e00-\u9fff\u0041-\u005a\u0061-\u007a\u3000-\u303f\uff00-\uffef0-9a-z.,!?;:\'\"()（）。，！？；：""''【】《》\n ]', '', text)
        text = re.sub(r'\s{3,}', '  ', text)
        return text.strip()

    def chat_stream(self, user_input: str) -> str:
        payload = self._build_payload(user_input)

        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                stream=True,
                timeout=300,
            )
            resp.raise_for_status()

            full_text = ""
            reasoning_buffer = ""
            has_printed_reasoning_newline = False

            for line in resp.iter_lines():
                if not line:
                    continue
                line = line.decode("utf-8", errors="ignore")
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    break

                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                choices = chunk.get("choices", [])
                if not choices:
                    continue

                delta = choices[0].get("delta", {})
                c_chunk = delta.get("content") or ""
                r_chunk = delta.get("reasoning_content") or ""

                # 处理 reasoning 内容
                if r_chunk:
                    reasoning_buffer += r_chunk
                    if self.show_reasoning:
                        # 过滤并显示有意义的文本
                        printable = r_chunk.strip()
                        if printable and len(printable) > 0:
                            # 检查是否包含可读字符（中文、英文等）
                            readable = any(
                                '\u4e00' <= ch <= '\u9fff' or
                                '\u0041' <= ch <= '\u005a' or
                                '\u0061' <= ch <= '\u007a'
                                for ch in printable
                            )
                            if readable:
                                if not has_printed_reasoning_newline:
                                    print("", flush=True)
                                    has_printed_reasoning_newline = True
                                print(f"{C.DIM}{printable}{C.RESET}", end="", flush=True)

                # 处理正式回复内容
                if c_chunk:
                    full_text += c_chunk
                    print(c_chunk, end="", flush=True)

            if full_text or reasoning_buffer:
                print()

            final_reply = full_text if full_text else self._clean_text(reasoning_buffer)

            self.add_message("user", user_input)
            self.add_message("assistant", final_reply)
            return final_reply

        except requests.exceptions.Timeout:
            print(f"\n{C.RED}⚠️ 请求超时{C.RESET}")
            return ""
        except requests.exceptions.ConnectionError as e:
            print(f"\n{C.RED}❌ 连接失败: {e}{C.RESET}")
            return ""
        except Exception as e:
            print(f"\n{C.RED}❌ 错误: {e}{C.RESET}")
            return ""

    def chat_nonstream(self, user_input: str) -> str:
        payload = self._build_payload(user_input)
        payload["stream"] = False

        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                timeout=300,
            )
            resp.raise_for_status()
            data = resp.json()

            choice = data.get("choices", [{}])[0]
            message = choice.get("message", {})
            content = message.get("content") or ""
            reasoning = message.get("reasoning_content") or ""

            if reasoning and self.show_reasoning:
                cleaned = self._clean_text(reasoning)
                if cleaned:
                    print(f"{C.DIM}{cleaned}{C.RESET}")

            if content:
                print(content)
            elif reasoning:
                print(self._clean_text(reasoning))

            final_reply = content if content else self._clean_text(reasoning)
            self.add_message("user", user_input)
            self.add_message("assistant", final_reply)

            usage = data.get("usage", {})
            if usage:
                self._update_usage(usage)
            return final_reply

        except Exception as e:
            print(f"{C.RED}❌ 错误: {e}{C.RESET}")
            return ""

    def _update_usage(self, usage: dict):
        self.token_count["prompt"] += usage.get("prompt_tokens", 0)
        self.token_count["completion"] += usage.get("completion_tokens", 0)
        self.token_count["total"] += usage.get("total_tokens", 0)

    # ---- 会话管理 ----

    def save_session(self, filename: str = None):
        os.makedirs(SAVE_DIR, exist_ok=True)
        if filename is None:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chat_{ts}.json"
        filepath = os.path.join(SAVE_DIR, filename)
        session_data = {
            "model": self.model,
            "created_at": datetime.datetime.now().isoformat(),
            "turns": self.turn_count,
            "messages": self.messages,
            "token_usage": self.token_count,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(session_data, f, ensure_ascii=False, indent=2)
        print(f"{C.GREEN}💾 已保存: {filepath}{C.RESET}")

    def load_session(self, filepath: str):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.messages = data.get("messages", [])
        self.token_count = data.get("token_usage", {"prompt": 0, "completion": 0, "total": 0})
        print(f"{C.GREEN}📂 已加载: {filepath} ({len(self.messages)} 条消息){C.RESET}")

    def list_saved_sessions(self) -> list[str]:
        if not os.path.exists(SAVE_DIR):
            return []
        return sorted(
            [f for f in os.listdir(SAVE_DIR) if f.endswith(".json")],
            key=lambda f: os.path.getmtime(os.path.join(SAVE_DIR, f)),
            reverse=True,
        )


# ============================================================
# UI
# ============================================================

def print_banner():
    print()
    print(f"{'='*60}")
    print(f"  {C.BOLD}{C.CYAN}🤖 Qwen3.5-35B-A3B 对话终端{C.RESET}")
    print(f"  {C.DIM}SGLang: http://33.29.18.248:44400{C.RESET}")
    print(f"{'='*60}")
    print(f"  {C.YELLOW}命令:{C.RESET} /clear /quit /mode /reason /temp /maxtok /tokens /save /load /help")
    print()


def print_help():
    print(f"""
{C.BOLD}命令列表:{C.RESET}
  {C.CYAN}/clear{C.RESET}              清空上下文
  {C.CYAN}/quit{C.EXIT}               退出
  {C.CYAN}/mode stream|normal{C.RESET}   切换流式/普通模式
  {C.CYAN}/reason on|off{C.RESET}       开关思维链显示
  {C.CYAN}/temp <0-2>{C.RESET}           设置温度 (默认 0.7)
  {C.CYAN}/maxtok <N>{C.RESET}           设置最大 Token (默认 4096)
  {C.CYAN}/tokens{C.RESET}             Token 统计
  {C.CYAN}/save [name]{C.RESET}          保存对话
  {C.CYAN}/load{C.RESET}               加载历史对话
  {C.CYAN}/info{C.RESET}               当前配置
  {C.CYAN}/help{C.RESET}               帮助
""")


def print_info(s: ChatSession):
    print(f"""
{C.BOLD}配置信息:{C.RESET}
   模型:     Qwen3.5-35B-A3B (MoE)
   服务端:   33.29.18.248:44400
  模式:     {'流式' if s.stream_mode else '普通'}
  思维链:   {'显示' if s.show_reasoning else '隐藏'}
  温度:     {s.temperature}
  MaxToken: {s.max_tokens}
  对话轮次: {s.turn_count}
  Token:    P={s.token_count['prompt']} C={s.token_count['completion']} T={s.token_count['total']}
""")


# ============================================================
# 主循环
# ============================================================

def main():
    session = ChatSession(base_url=SGLANG_BASE_URL, model=MODEL_ID)
    print_banner()

    while True:
        try:
            user_input = input(f"{C.GREEN}You{C.RESET}{C.BOLD}:{C.RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{C.YELLOW}再见！{C.RESET}")
            break

        if not user_input:
            continue

        # ---- 命令处理 ----
        if user_input.startswith("/"):
            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else None

            if cmd in ("/quit", "/exit", "/q"):
                print(f"{C.YELLOW}再见！{C.RESET}")
                break
            elif cmd == "/clear":
                session.clear()
            elif cmd == "/help":
                print_help()
            elif cmd == "/info":
                print_info(session)
            elif cmd == "/mode":
                if arg == "normal":
                    session.stream_mode = False; print(f"{C.CYAN}→ 普通模式{C.RESET}")
                elif arg == "stream":
                    session.stream_mode = True; print(f"{C.CYAN}→ 流式模式{C.RESET}")
                else:
                    print(f"{C.CYAN}当前: {'流式' if session.stream_mode else '普通'}{C.RESET}")
            elif cmd == "/reason":
                if arg == "off":
                    session.show_reasoning = False; print(f"{C.CYAN}→ 隐藏思维链{C.RESET}")
                elif arg == "on":
                    session.show_reasoning = True; print(f"{C.CYAN}→ 显示思维链{C.RESET}")
                else:
                    print(f"{C.CYAN}思维链: {'显示' if session.show_reasoning else '隐藏'}{C.RESET}")
            elif cmd == "/temp":
                if arg:
                    try:
                        v = float(arg)
                        if 0 <= v <= 2:
                            session.temperature = v; print(f"{C.CYAN}温度={v}{C.RESET}")
                        else:
                            print(f"{C.RED}范围 0~2{C.RESET}")
                    except ValueError:
                        print(f"{C.RED}无效值{C.RESET}")
                else:
                    print(f"{C.CYAN}温度={session.temperature}{C.RESET}")
            elif cmd == "/maxtok":
                if arg:
                    try:
                        v = int(arg)
                        if v > 0:
                            session.max_tokens = v; print(f"{C.CYAN}MaxTokens={v}{C.RESET}")
                        else:
                            print(f"{C.RED}>0{C.RESET}")
                    except ValueError:
                        print(f"{C.RED}无效值{C.RESET}")
                else:
                    print(f"{C.CYAN}MaxTokens={session.max_tokens}{C.RESET}")
            elif cmd == "/tokens":
                t = session.token_count
                print(f"{C.BOLD}Token统计:{C.RESET}\n  Prompt:{t['prompt']:,}  Completion:{t['completion']:,}  Total:{t['total']:,}  轮次:{session.turn_count}")
            elif cmd == "/save":
                session.save_session(arg)
            elif cmd == "/load":
                files = session.list_saved_sessions()
                if not files:
                    print(f"{C.YELLOW}无保存记录{C.RESET}"); continue
                print(f"\n{C.BOLD}历史对话:{C.RESET}")
                for i, f in enumerate(files):
                    fp = os.path.join(SAVE_DIR, f)
                    try:
                        with open(fp) as fh:
                            d = json.load(fh)
                        turns = d.get("turns", "?")
                        date = (d.get("created_at") or "")[:16]
                        preview = ""
                        for m in d.get("messages", []):
                            if m.get("role") == "user" and m.get("content"):
                                preview = m["content"][:40].replace("\n", " "); break
                        print(f"  {C.CYAN}[{i}]{C.RESET} {f}  ({turns}轮, {date})")
                        if preview:
                            print(f"       {C.DIM}{preview}{C.RESET}")
                    except:
                        print(f"  {C.CYAN}[{i}]{C.RESET} {f}")
                sel = input(f"\n{C.YELLOW}选择编号 (回车取消):{C.RESET} ").strip()
                if sel.isdigit() and 0 <= int(sel) < len(files):
                    session.load_session(os.path.join(SAVE_DIR, files[int(sel)]))
            else:
                print(f"{C.RED}未知: {cmd} (/help){C.RESET}")
            continue

        # ---- 对话 ----
        print(f"\n{C.MAGENTA}Assistant{C.RESET}: ", end="")
        if session.stream_mode:
            session.chat_stream(user_input)
        else:
            session.chat_nonstream(user_input)
        print()


if __name__ == "__main__":
    main()

