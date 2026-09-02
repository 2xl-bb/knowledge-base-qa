"""公共工具：读 API Key、调用模型、算向量、算相似度

本项目的两个服务商，各有一个 key 文件：
- key.txt     智谱（bigmodel.cn）：对话模型 glm-4-flash，免费
- key_sf.txt  硅基流动（siliconflow.cn）：向量模型 BAAI/bge-m3，免费
"""
import json
import os
import sys
import urllib.error
import urllib.request

# 让中文在 Windows 终端里正常显示
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass

# 程序自己所在的文件夹，用它来定位 key 文件，这样在任何目录运行都能找到
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

ZHIPU_URL_CHAT = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
SF_URL_EMBED = "https://api.siliconflow.cn/v1/embeddings"


def read_key(filename):
    """读指定 key 文件。不同软件保存的编码可能不一样，几种常见编码都试一次"""
    for enc in ("utf-8-sig", "utf-16", "gbk"):
        try:
            with open(os.path.join(SCRIPT_DIR, filename), encoding=enc) as f:
                return f.read().strip()
        except (FileNotFoundError, UnicodeDecodeError):
            continue
    return ""


API_KEY = read_key("key.txt")
SF_API_KEY = read_key("key_sf.txt")

if not API_KEY:
    print("还没有智谱 API Key：在项目文件夹创建 key.txt，把 open.bigmodel.cn 的 Key 粘贴进去。")
    sys.exit(1)
if not SF_API_KEY:
    print("还没有硅基流动 API Key（向量要用）：")
    print("1. 打开 https://siliconflow.cn 用手机号注册")
    print("2. 进控制台 -> API 密钥 -> 创建新密钥")
    print("3. 在项目文件夹创建 key_sf.txt，把密钥粘贴进去保存")
    sys.exit(1)


def _post(url: str, payload: dict, api_key: str) -> dict:
    """给 API 发一个 POST 请求，返回解析后的 JSON"""
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": "Bearer " + api_key,
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print("请求失败了，HTTP 状态码：", e.code)
        print("服务器说明：", body[:500])
        print("429 = 请求太频繁被限流，稍等几秒重试；401 = API Key 不对，检查 key 文件。")
        sys.exit(1)


def call_llm(system: str, user: str) -> str:
    """跟大模型对话（智谱 glm-4-flash），返回回答"""
    data = _post(
        ZHIPU_URL_CHAT,
        {
            "model": "glm-4-flash",  # 免费模型
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        },
        API_KEY,
    )
    return data["choices"][0]["message"]["content"]


def embed_many(texts: list, batch_size: int = 64) -> list:
    """把多段文字变成向量（硅基流动 BAAI/bge-m3），返回顺序与传入一致。

    接口单次请求能接收的文本条数有限制，段数多了就分几批发，
    每批最多 batch_size 条（64 是常见的稳妥上限）。"""
    results = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        data = _post(
            SF_URL_EMBED,
            {
                "model": "BAAI/bge-m3",  # 免费向量模型，1024 维
                "input": batch,
            },
            SF_API_KEY,
        )
        for item in sorted(data["data"], key=lambda x: x["index"]):
            results.append(item["embedding"])
    return results


def embed(text: str) -> list:
    """把一段文字变成向量（一串数字）。语义相近的文字，向量也相近"""
    return embed_many([text])[0]


def cosine_similarity(a: list, b: list) -> float:
    """两个向量的余弦相似度：越接近 1 表示越相似"""
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb)
