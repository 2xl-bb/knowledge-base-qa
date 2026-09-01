"""第 3 步：知识库问答的网页版

运行：
    streamlit run app.py

浏览器会自动打开。输入问题，程序会先检索笔记，再让模型回答，
并且展示"这段回答用到了哪些笔记"——这就是 RAG 的可追溯性。
"""
import json

import streamlit as st

import common

st.set_page_config(page_title="我的课程笔记知识库")

# 读取知识库（先运行过 python build_kb.py 才会有 kb.json）
try:
    with open("kb.json", encoding="utf-8") as f:
        kb = json.load(f)
except FileNotFoundError:
    st.error("还没找到 kb.json！请先在终端运行：python build_kb.py 来建库。")
    st.stop()

st.title("我的课程笔记知识库")
st.caption("原理：先检索最相关的笔记片段，再让大模型基于笔记回答")

# 侧边栏：知识库概况
with st.sidebar:
    st.subheader("知识库概况")
    files = sorted({src.split("（")[0] for src in kb["sources"]})
    st.write(f"共 {len(kb['chunks'])} 段笔记，来自 {len(files)} 个文件：")
    for name in files:
        st.write(f"- {name}")
    show_sources = st.checkbox("显示检索到的笔记原文", value=True)

question = st.text_input("你的问题", placeholder="例如：朴素贝叶斯为什么适合垃圾邮件过滤？")

if question:
    with st.spinner("正在检索笔记……"):
        q_vec = common.embed(question)
        scored = []
        for vec, src, chunk in zip(kb["vectors"], kb["sources"], kb["chunks"]):
            scored.append((common.cosine_similarity(q_vec, vec), src, chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:4]

    context = "\n\n".join(f"【{src}】\n{chunk}" for _, src, chunk in top)
    with st.spinner("正在让模型回答……"):
        answer = common.call_llm(
            "你是知识库问答助手。请只根据用户提供的笔记内容回答问题，"
            "笔记里没有的内容，就如实说'笔记里没找到'。用中文回答，简洁一点。",
            f"笔记内容：\n{context}\n\n问题：{question}",
        )

    st.success(answer)

    if show_sources:
        st.divider()
        st.subheader("这段回答用到的笔记")
        for score, src, chunk in top:
            with st.expander(f"{src}（相似度 {score:.3f}）"):
                st.write(chunk)
