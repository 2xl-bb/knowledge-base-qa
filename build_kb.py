"""第 2 步（建库）：把 notes 文件夹里的笔记建成知识库

运行：
    python build_kb.py

流程：读取 notes/ 下的 .txt、.md 和 .pdf（有文字层的）文件 ->
切成长短合适的块 -> 每块用 API 算成向量 -> 存进 kb.json。
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


def extract_pdf_pages(path: str) -> list:
    """提取 PDF 每页的文字，返回 [(页码, 该页文字)]。
    只支持有文字层的 PDF（课件/论文这类数字导出的）；扫描件没有文字层，返回空列表。"""
    try:
        from pypdf import PdfReader
    except ImportError:
        print("缺少 pypdf 库，无法读 PDF：先运行  python -m pip install pypdf")
        raise SystemExit(1)
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append((i, text))
    return pages


def load_notes(folder="notes"):
    """读取 notes/ 下的笔记文件。
    - .txt / .md：整个文件作为一个来源
    - .pdf：逐页提取，每页作为一个来源（来源名带页码，方便追溯回答出自哪页）
    返回 [(来源名, 文本)]。"""
    notes = []
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if name.endswith((".txt", ".md")):
            text = read_text(path)
            if text.strip():
                notes.append((name, text))
        elif name.lower().endswith(".pdf"):
            pages = extract_pdf_pages(path)
            if not pages:
                print(f"  ! 跳过 {name}：提取不到文字，可能是扫描件（需要 OCR）")
                continue
            for pageno, page_text in pages:
                notes.append((f"{name} 第{pageno}页", page_text))
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
        print("把 .txt / .md 课程笔记，或有文字层的 .pdf 课件放进去，再运行本脚本。")
        return

    print(f"共 {len(notes)} 个来源（文件/页面），开始切块……")
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
