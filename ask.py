"""第 2 步（问答）：向知识库提问

运行：
    python ask.py "你的问题"

流程：把你的问题也变成向量 -> 在 kb.json 里找出最相关的几段笔记 ->
把这几段原文和问题一起交给大模型 -> 大模型只根据笔记内容回答。
"""
import json
import sys

import common


def main():
    try:
        with open("kb.json", encoding="utf-8") as f:
            kb = json.load(f)
    except FileNotFoundError:
        print("还没找到 kb.json！请先运行：python build_kb.py 来建库。")
        return

    question = " ".join(sys.argv[1:]) or "知识库里讲了什么？"
    print("问题：", question)
    print("正在计算相似度，找出最相关的笔记……")
    q_vec = common.embed(question)

    # 每个笔记块和问题算一个相似度，按相似度从高到低排序
    scored = []
    for vec, src, chunk in zip(kb["vectors"], kb["sources"], kb["chunks"]):
        scored.append((common.cosine_similarity(q_vec, vec), src, chunk))
    scored.sort(key=lambda x: x[0], reverse=True)

    top = scored[:4]
    print(f"找到最相关的 {len(top)} 段笔记：")
    for score, src, chunk in top:
        print(f"- [{src}] 相似度 {score:.3f} | {chunk[:40]}……")
    print()

    # 把笔记原文拼成上下文，交给大模型
    context = "\n\n".join(f"【{src}】\n{chunk}" for _, src, chunk in top)
    answer = common.call_llm(
        "你是知识库问答助手。请只根据用户提供的笔记内容回答问题，"
        "笔记里没有的内容，就如实说'笔记里没找到'。用中文回答，简洁一点。",
        f"笔记内容：\n{context}\n\n问题：{question}",
    )
    print("回答：", answer)


if __name__ == "__main__":
    main()
