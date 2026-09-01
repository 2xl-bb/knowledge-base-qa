"""知识库问答 · 第 1 步：跟大模型对话

用法：
    python chat.py "你的问题"

提示词（人设）也可以在这里改。
"""
import sys

import common

if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "你好，请用三句话介绍你自己"
    print("你问：", question)
    print()
    print("模型答：", common.call_llm("你是南邮的师兄，回答要接地气", question))
