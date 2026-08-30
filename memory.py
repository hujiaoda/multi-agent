"""记忆层：把每次运行的经验存进 memory/*.json，新需求进来时用 embedding 检索最相关的历史。

设计（延续"文件系统是真相"）：
- 存：每次运行一条 JSON，向量就存在 JSON 里（embedding 只对"需求+方案摘要"生成）
- 查：新需求 embedding → 遍历所有记录算余弦相似度 → top_k 最相关的
- 规模小（几十~几百条），暴力遍历够用；量大再换向量数据库
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import json
import math
from datetime import datetime
from pathlib import Path

from openai import OpenAI

from config import QWEN_API_KEY, QWEN_BASE_URL

MEMORY_DIR = Path("memory")
EMBED_MODEL = "text-embedding-v3"

_client = None


def _get_client() -> OpenAI:
    """懒加载 OpenAI 客户端（和 rag_demo 同一套 dashscope 兼容接口）"""
    global _client
    if _client is None:
        _client = OpenAI(api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL)
    return _client


def embed_texts(texts: list[str]) -> list[list[float]]:
    """把文本列表变成向量列表（按传入顺序返回）"""
    resp = _get_client().embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]


def cosine_sim(a: list[float], b: list[float]) -> float:
    """两个向量的余弦相似度（-1~1，越大越像）"""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def save_run(run_id: str, requirement: str, plan_summary: str = "", files=None,
             result: str = "", pitfall: str = "", workdir: str = "") -> str:
    """运行结束存一条记忆。embedding 只对"需求+方案摘要"生成（检索用），代码本体不复制。"""
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    # 检索文本 = 需求 + 方案摘要 + 踩坑（踩坑截断，避免长串失败信息当噪声）
    search_text = f"{requirement}\n{plan_summary}\n{(pitfall or '')[:100]}"
    embedding = embed_texts([search_text])[0]
    record = {
        "run_id": run_id,
        "requirement": requirement,
        "plan_summary": plan_summary,
        "files": files or [],
        "result": result,
        "pitfall": pitfall,
        "workdir": workdir,
        "embedding": embedding,
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    path = MEMORY_DIR / f"{run_id}.json"
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def search_memory(requirement: str, top_k: int = 3) -> list[dict]:
    """新需求进来：embedding + 余弦，返回相似度最高的 top_k 条记录（供参考）。"""
    if not MEMORY_DIR.exists():
        return []
    records = [json.loads(p.read_text(encoding="utf-8")) for p in MEMORY_DIR.glob("*.json")]
    if not records:
        return []
    query_vec = embed_texts([requirement])[0]
    scored = [
        (cosine_sim(query_vec, rec["embedding"]), rec)
        for rec in records
        if rec.get("embedding")
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [rec for _, rec in scored[:top_k]]


def format_memory_hits(hits: list[dict]) -> str:
    """把检索结果格式化成给模型看的历史经验文本（每条都截断，控制上下文量）"""
    lines = []
    for i, rec in enumerate(hits, 1):
        source = rec.get("workdir") or f"memory/{rec.get('run_id')}.json"
        lines.append(
            f"[历史 {i}] 需求：{rec.get('requirement', '')[:100]}\n"
            f"方案：{rec.get('plan_summary', '')[:100]}\n"
            f"结果：{rec.get('result', '')}\n"
            f"踩坑：{rec.get('pitfall', '')[:100]}\n"
            f"来源：{source}"
        )
    return "\n\n".join(lines)
