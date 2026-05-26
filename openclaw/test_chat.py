"""Final test: Qwen3.5-35B-A3B with chat program"""
import sys
sys.path.insert(0, '.')

from chat import ChatSession, C

session = ChatSession(
    base_url='http://33.29.18.248:44400/v1',
    model='/home/hadoop-djst-algoplat/model/Qwen/Qwen3.5-35B-A3B'
)
session.stream_mode = True
session.max_tokens = 8192

print(f'{C.BOLD}Qwen3.5-35B-A3B 对话测试{C.RESET}')
print()

# Test 1: Self-intro
print(f'{C.GREEN}You{C.RESET}: 你好！请简短自我介绍')
print(f'{C.MAGENTA}Assistant{C.RESET}: ', end='')
r1 = session.chat_stream('你好！请用中文简短自我介绍，2-3句话即可')
print(f'\n{C.DIM}(回复长度: {len(r1)} 字符){C.RESET}\n')

# Test 2: Math
print(f'{C.GREEN}You{C.RESET}: 1+1等于几？只回答数字')
print(f'{C.MAGENTA}Assistant{C.RESET}: ', end='')
r2 = session.chat_stream('1+1等于几？只回答数字')
print(f'\n{C.DIM}(回复长度: {len(r2)} 字符){C.RESET}\n')

# Test 3: Code
print(f'{C.GREEN}You{C.RESET}: 用Python写一个快速排序')
print(f'{C.MAGENTA}Assistant{C.RESET}: ', end='')
r3 = session.chat_stream('用Python写一个快速排序函数')
print(f'\n{C.DIM}(回复长度: {len(r3)} 字符){C.RESET}\n')

# Summary
print(f'{C.BOLD}=== 测试完成 ==={C.RESET}')
print(f'总对话轮数: {session.turn_count}')
t = session.token_count
print(f'Token: Prompt={t["prompt"]:,} Completion={t["completion"]:,} Total={t["total"]:,}')
