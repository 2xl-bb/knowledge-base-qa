"""第 2 步（建库）：把 notes 文件夹里的笔记建成知识库

运行：
    python build_kb.py

流程：读取 notes/ 下的 .txt 和 .md 文件 -> 切成长短合适的块 ->
每块用 API 算成向量 -> 存进 kb.json。
以后笔记更新了，重新运行一次即可。
"""
import json
import os
import re

import common


def read_text(path: str) -> str:
    """读文件，自动适配 UTF-8 / GBK 编码"""
    for enc in ("utf-8-sig", "gbk"):
        try:
            with open(path, encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    return ""


def load_notes(folder="notes"):
    """读取文件夹里所有 .txt 和 .md 文件，返回 [(文件名, 全文)]"""
    notes = []
    for name in sorted(os.listdir(folder)):
        if name.endswith((".txt", ".md")):
            text = read_text(os.path.join(folder, name))
            if text.strip():
                notes.append((name, text))
    return notes


def chunk_text(text: str, size: int = 400, overlap: int = 50) -> list:
    """把长文本切成小块。每块最多 size 个字符，相邻两块重叠 overlap 个字符，
    并尽量在句号、感叹号等句子结束的地方切断。"""
    text = re.sub(r"\s+", " ", text).strip()
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:  # 不是最后一块，优先在句子结尾处切断
            cut = -1
            for c in "。！？；":
                pos = text.rfind(c, start + size // 2, end)
                if pos > cut:
                    cut = pos
            if cut != -1:
                end = cut + 1
        chunks.append(text[start:end])
        start = end - overlap if end < n else n
    return [c for c in chunks if c.strip()]


def main():
    notes = load_notes()
    if not notes:
        print("notes 文件夹里还没有文件！")
        print("把 .txt 或 .md 格式的课程笔记放进去，再运行本脚本。")
        return

    print(f"找到 {len(notes)} 个文件，开始切块……")
    chunks = []
    sources = []
    for name, text in notes:
        parts = chunk_text(text)
        for i, part in enumerate(parts):
            chunks.append(part)
            sources.append(f"{name}（第{i + 1}段）")
        print(f"  {name}: {len(parts)} 段")

    print(f"共 {len(chunks)} 段，开始调用 API 计算向量（一次请求打包全部）……")
    vectors = common.embed_many(chunks)

    with open("kb.json", "w", encoding="utf-8") as f:
        json.dump(
            {"chunks": chunks, "sources": sources, "vectors": vectors},
            f,
            ensure_ascii=False,
        )
    print(f"知识库已保存到 kb.json（共 {len(chunks)} 段）。")


if __name__ == "__main__":
    main()
