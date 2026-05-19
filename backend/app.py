import os
os.environ["OMP_NUM_THREADS"] = "1"

import math
import json
import pickle
import shutil
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import torch
from functools import lru_cache

# =========================
# FastAPI
# =========================

app = FastAPI(title="STAR-SGG Demo Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# Paths
# =========================

ROOT = Path("/root/autodl-tmp/star-sgg-demo")
UPLOAD_DIR = ROOT / "uploads"
OUTPUT_DIR = ROOT / "outputs"
KG_RAG_RESULTS_PATH = ROOT / "star_kg_rag_results2.json"
LLM_MODEL_PATH = "/root/autodl-tmp/Qwen3-VL-8B-Instruct"

# 你的 SGG pickle 目录
SGG_DIR = Path("/root/autodl-tmp/Accurate_VQA-main/experiment/STAR_pre")

# 测试图片目录，用于根据文件名找到 prediction index
TEST_IMAGE_DIR = Path("/root/autodl-tmp/Accurate_VQA-main/experiment/STAR_test_images")

# 真实检测框结果文件
EVAL_RESULTS_PATH = Path(
    "/root/autodl-tmp/SGG_ToolKit_main/test_results/Final_Reproduce_SGDet/eval_results.pytorch"
)

# 类别字典，可选，用于把 label id 转成类别名
DICT_PATH = Path(
    "/root/autodl-tmp/SGG_ToolKit_main/STAR_dataset/STAR_SGG/STAR-SGG-dicts-with-attri.json"
)

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")


# =========================
# Basic utils
# =========================

def extract_image_id_from_filename(filename: str) -> int:
    """
    支持:
    0004.png -> 4
    0004__512__xxx.png -> 4
    """
    stem = Path(filename).stem
    prefix = stem.split("__")[0]
    return int(prefix)


def find_sgg_pickle(image_id: int):
    candidates = [
        SGG_DIR / f"{image_id:04d}.pickle",
        SGG_DIR / f"{image_id}.pickle",
    ]

    for p in candidates:
        if p.exists():
            return p

    return None


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


@lru_cache(maxsize=1)
def get_text_llm():
    from transformers import AutoProcessor, AutoModelForImageTextToText

    print(f"[LLM] Loading local model from: {LLM_MODEL_PATH}")

    processor = AutoProcessor.from_pretrained(
        LLM_MODEL_PATH,
        trust_remote_code=True,
    )

    model = AutoModelForImageTextToText.from_pretrained(
        LLM_MODEL_PATH,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )

    model.eval()

    print("[LLM] Model loaded successfully.")
    return processor, model


def call_text_llm(prompt: str, max_new_tokens: int = 256) -> str:
    import requests

    resp = requests.post(
        "http://127.0.0.1:6010/generate",
        json={
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
        },
        timeout=300,
    )

    resp.raise_for_status()
    data = resp.json()
    return data.get("answer", "").strip()

@lru_cache(maxsize=1)
def load_label_map():
    """
    尝试读取 STAR label 字典，把 pred_labels 转成类别名。
    如果读取失败，就返回空字典，前端会显示 cls_x。
    """
    if not DICT_PATH.exists():
        return {}

    try:
        with open(DICT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        idx_to_label = data.get("idx_to_label", {})
        return {int(k): v for k, v in idx_to_label.items()}
    except Exception as e:
        print(f"[WARN] failed to load label map: {e}")
        return {}

@lru_cache(maxsize=1)
def load_kg_rag_results():
    if not KG_RAG_RESULTS_PATH.exists():
        raise FileNotFoundError(f"KG-RAG results file not found: {KG_RAG_RESULTS_PATH}")

    with open(KG_RAG_RESULTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data.get("results", [])


def find_first_kg_rag_result(image_id=None, question_id=None):
    results = load_kg_rag_results()

    # 优先按 question_id 精确查找
    if question_id is not None:
        for item in results:
            if int(item.get("question_id", -1)) == int(question_id):
                return item

    # 没有 question_id，就按 image_id 查找第一条
    if image_id is not None:
        for item in results:
            if int(item.get("image_id", -1)) == int(image_id):
                return item

    return None

import re


def extract_option_letter(text: str) -> str:
    """
    从模型输出中提取 A/B/C/D。
    支持:
    B
    Answer: B
    The answer is B.
    """
    text = str(text or "").strip()

    if len(text) == 1 and text.upper() in ["A", "B", "C", "D"]:
        return text.upper()

    m = re.search(r"\b([A-D])\b", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()

    return ""


def parse_options_from_question(question_text: str) -> dict:
    """
    从 question 文本里提取选项句子。

    例如:
    A. The airplane is isolatedly parked on the apron
    B. The airplane is parallelly parked on the apron
    C. The airplane is randomly parked on the apron
    D. The airplane is over the apron.

    返回:
    {
      "A": "The airplane is isolatedly parked on the apron",
      "B": "The airplane is parallelly parked on the apron",
      ...
    }
    """
    text = str(question_text or "")

    pattern = re.compile(
        r"(?ms)^\s*([A-D])\.\s*(.*?)(?=^\s*[A-D]\.\s*|\Z)"
    )

    options = {}

    for m in pattern.finditer(text):
        letter = m.group(1).upper()
        content = m.group(2).strip()

        content = re.sub(
            r"\s*Answer with the option.*$",
            "",
            content,
            flags=re.IGNORECASE | re.DOTALL,
        ).strip()

        options[letter] = content

    return options

# =========================
# eval_results.pytorch loading
# =========================

@lru_cache(maxsize=1)
def load_eval_predictions():
    print(f"[INFO] loading eval results from: {EVAL_RESULTS_PATH}")
    data = torch.load(str(EVAL_RESULTS_PATH), map_location="cpu")
    return data["predictions"]


@lru_cache(maxsize=1)
def load_test_image_names():
    image_files = sorted(
        [
            p.name
            for p in TEST_IMAGE_DIR.iterdir()
            if p.suffix.lower() in [".png", ".jpg", ".jpeg"]
        ]
    )
    return image_files


def get_prediction_index_by_filename(filename: str) -> int:
    """
    通过上传图片文件名，在 TEST_IMAGE_DIR 排序后的列表中找到对应 index。
    注意：这要求 eval_results.pytorch 的 predictions 顺序和 TEST_IMAGE_DIR 图片顺序一致。
    """
    image_name = Path(filename).name
    image_files = load_test_image_names()

    if image_name not in image_files:
        raise FileNotFoundError(
            f"{image_name} not found in {TEST_IMAGE_DIR}. "
            f"Make sure uploaded filename matches test image filename."
        )

    return image_files.index(image_name)


def get_prediction_by_filename(filename: str):
    predictions = load_eval_predictions()
    idx = get_prediction_index_by_filename(filename)

    if idx >= len(predictions):
        raise IndexError(
            f"Prediction index {idx} out of range. total predictions={len(predictions)}"
        )

    return predictions[idx], idx


# =========================
# pickle normalization
# =========================

def normalize_pre_data(data, max_relations_for_demo=None):
    """
    演示用：
    1. raw_relation_count 记录原始关系总数
    2. relationship 只保留去重后的类别级关系，保证结构图清晰
    """

    if not isinstance(data, dict):
        return {
            "number": {},
            "location": {},
            "relationship": [],
            "raw_relation_count": 0,
            "display_relation_count": 0,
        }

    number = data.get("number", {})
    location = data.get("location", {})
    relationship = data.get("relationship", [])

    raw_relation_count = len(relationship) if isinstance(relationship, list) else 0

    if not isinstance(number, dict):
        number = {}

    if not isinstance(location, dict):
        location = {}

    clean_rels = []
    seen = set()

    if isinstance(relationship, list):
        for rel in relationship:
            if not isinstance(rel, (list, tuple)) or len(rel) != 3:
                continue

            subj, pred, obj = rel
            subj = str(subj)
            pred = str(pred)
            obj = str(obj)

            key = (subj, pred, obj)
            if key in seen:
                continue

            seen.add(key)
            clean_rels.append([subj, pred, obj])

            if max_relations_for_demo is not None and len(clean_rels) >= max_relations_for_demo:
                break

    return {
        "number": number,
        "location": location,
        "relationship": clean_rels,
        "raw_relation_count": raw_relation_count,
        "display_relation_count": len(clean_rels),
    }

def build_interactive_graph(pre_data):
    number = pre_data.get("number", {})
    relationship = pre_data.get("relationship", [])

    objects = []
    for name, count in number.items():
        objects.append({
            "id": str(name),
            "name": str(name),
            "label": str(name),
            "count": int(count),
        })

    relations = []
    seen = set()

    for rel in relationship:
        if not isinstance(rel, (list, tuple)) or len(rel) != 3:
            continue

        subj, pred, obj = map(str, rel)

        # 去重：相同 subject-predicate-object 只画一次
        key = (subj, pred, obj)
        if key in seen:
            continue
        seen.add(key)

        relations.append({
            "source": subj,
            "target": obj,
            "relation": pred,
            "label": pred,
        })

    return {
        "objects": objects,
        "relations": relations,
    }

def sgg_to_text(pre_data, image_name=""):
    number = pre_data.get("number", {})
    location = pre_data.get("location", {})
    relationship = pre_data.get("relationship", [])

    categories = list(number.keys())

    lines = []

    if image_name:
        lines.append(f"Image: {image_name}")
        lines.append("")

    lines.append("Category:")
    if categories:
        lines.append(", ".join(categories))
    else:
        lines.append("None")

    lines.append("")
    lines.append("Quantity:")
    if number:
        for k, v in number.items():
            lines.append(f"- {k}: {v}")
    else:
        lines.append("None")

    lines.append("")
    lines.append("Location:")
    if location:
        for obj, locs in location.items():
            if isinstance(locs, list):
                loc_preview = locs[:40]
                loc_text = ", ".join(map(str, loc_preview))
                if len(locs) > 40:
                    loc_text += f", ... total {len(locs)}"
                lines.append(f"- {obj}: {loc_text}")
            else:
                lines.append(f"- {obj}: {locs}")
    else:
        lines.append("None")

    lines.append("")
    lines.append("Relationship:")
    if relationship:
        max_show = 150
        for rel in relationship[:max_show]:
            if isinstance(rel, (list, tuple)) and len(rel) == 3:
                subj, pred, obj = map(str, rel)
                lines.append(f"- {subj} -- {pred} --> {obj}")
        if len(relationship) > max_show:
            lines.append(f"... total relationships: {len(relationship)}")
    else:
        lines.append("None")

    return "\n".join(lines)

# =========================
# Draw rotated detection boxes
# =========================

def rotated_box_to_points(cx, cy, w, h, angle):
    """
    5维旋转框转四点。
    你的 bbox shape 是 [N, 5]，一般可理解为:
    cx, cy, w, h, angle
    """

    corners = np.array(
        [
            [-w / 2, -h / 2],
            [ w / 2, -h / 2],
            [ w / 2,  h / 2],
            [-w / 2,  h / 2],
        ],
        dtype=np.float32,
    )

    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    rot = np.array(
        [
            [cos_a, -sin_a],
            [sin_a,  cos_a],
        ],
        dtype=np.float32,
    )

    rotated = corners @ rot.T
    rotated[:, 0] += cx
    rotated[:, 1] += cy

    return [(float(x), float(y)) for x, y in rotated]


def scale_rotated_box(box, pred_size, image_size):
    """
    如果预测结果对应的图像尺寸和上传图片尺寸不同，需要缩放。
    pred_size 通常是 (width, height)
    image_size 也是 (width, height)
    """

    if len(box) != 5:
        return box

    pred_w, pred_h = pred_size
    img_w, img_h = image_size

    sx = img_w / pred_w
    sy = img_h / pred_h

    cx, cy, w, h, angle = box.tolist()

    return np.array(
        [
            cx * sx,
            cy * sy,
            w * sx,
            h * sy,
            angle,
        ],
        dtype=np.float32,
    )


def draw_detection_boxes_on_image(
    image_path: Path,
    pred,
    output_path: Path,
    score_thresh: float = 0.0,
    max_boxes: int = 500,
):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    boxes = pred.bbox.cpu().numpy()

    labels = None
    scores = None

    if pred.has_field("pred_labels"):
        labels = pred.get_field("pred_labels").cpu().numpy()

    if pred.has_field("pred_scores"):
        scores = pred.get_field("pred_scores").cpu().numpy()

    label_map = load_label_map()

    indices = list(range(len(boxes)))

    if scores is not None:
        indices = [int(i) for i in np.argsort(-scores) if scores[i] >= score_thresh]

    indices = indices[:max_boxes]

    pred_size = getattr(pred, "size", image.size)
    image_size = image.size

    for i in indices:
        box = boxes[i]

        if len(box) == 5:
            scaled_box = scale_rotated_box(box, pred_size, image_size)
            cx, cy, w, h, angle = scaled_box.tolist()
            pts = rotated_box_to_points(cx, cy, w, h, angle)

            # 画闭合旋转框
            draw.line(pts + [pts[0]], fill="red", width=2)
            tx, ty = pts[0]

        else:
            x1, y1, x2, y2 = box[:4].tolist()
            pred_w, pred_h = pred_size
            img_w, img_h = image_size
            sx = img_w / pred_w
            sy = img_h / pred_h

            x1 *= sx
            x2 *= sx
            y1 *= sy
            y2 *= sy

            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            tx, ty = x1, y1

        label_text = "obj"

        if labels is not None:
            label_id = int(labels[i])
            label_text = label_map.get(label_id, f"cls_{label_id}")

        if scores is not None:
            label_text += f" {float(scores[i]):.2f}"

        draw.text((tx, max(0, ty - 13)), label_text, fill="red")

    image.save(output_path)


# =========================
# Draw SGG relation graph
# =========================



# =========================
# Draw SGG relation graph (胶囊图风格)
# =========================

def draw_scene_graph(pre_data, output_png):
    import math
    from collections import defaultdict

    import networkx as nx
    import matplotlib.pyplot as plt

    relationships = pre_data.get("relationship", [])
    number_info = pre_data.get("number", {})

    # ---------------------------
    # 1. MultiDiGraph：一条关系就是一条边
    # ---------------------------
    G = nx.MultiDiGraph()

    for rel in relationships:
        if not isinstance(rel, (list, tuple)) or len(rel) != 3:
            continue
        subj, pred, obj = map(str, rel)
        G.add_node(subj)
        G.add_node(obj)
        G.add_edge(subj, obj, relation=pred)

    for node in G.nodes():
        G.nodes[node]["count"] = int(number_info.get(node, 1))

    fig, ax = plt.subplots(figsize=(14, 9), dpi=220)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    if len(G.nodes()) == 0:
        ax.text(0.5, 0.5, "未检测到场景图关系", ha="center", va="center", fontsize=16)
        ax.axis("off")
        fig.savefig(output_png, bbox_inches="tight", pad_inches=0.3)
        plt.close(fig)
        return

    # ---------------------------
    # 2. 固定语义布局（你这个机场图比较适合）
    # ---------------------------
    preset_pos = {
        "tank": (-2.4, 1.8),
        "taxiway": (-1.7, 0.0),
        "apron": (-1.0, -1.4),
        "boarding_bridge": (0.6, 1.0),
        "airplane": (2.1, -0.1),
        "runway": (1.0, -1.7),
        "terminal": (1.4, 2.0),
    }

    pos = {}
    for node in G.nodes():
        if node in preset_pos:
            pos[node] = preset_pos[node]

    missing_nodes = [n for n in G.nodes() if n not in pos]
    if missing_nodes:
        subG = G.subgraph(missing_nodes).copy()
        auto_pos = nx.spring_layout(subG, seed=42, k=2.0)
        for n, p in auto_pos.items():
            pos[n] = p

    # ---------------------------
    # 3. 节点颜色
    # ---------------------------
    color_map = {
        "taxiway": "#C9D9F2",
        "runway": "#D4B2E8",
        "apron": "#CDEBC8",
        "airplane": "#F3D86D",
        "boarding_bridge": "#F7D8A2",
        "tank": "#F6C6C6",
        "terminal": "#E5E5E5"
    }

    # ---------------------------
    # 4. 节点大小：圆形大一点，包住文字
    # ---------------------------
    node_sizes = []
    for node in G.nodes():
        count = G.nodes[node].get("count", 1)
        text_len = len(f"{node} ({count})")
        size = max(5200, 3800 + text_len * 220)
        node_sizes.append(size)

    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=node_sizes,
        node_color=[color_map.get(n, "#DDDDDD") for n in G.nodes()],
        edgecolors="#333333",
        linewidths=1.6,
        node_shape="o",
        ax=ax
    )

    nx.draw_networkx_labels(
        G,
        pos,
        labels={n: f"{n} ({G.nodes[n].get('count', 1)})" for n in G.nodes()},
        font_size=12,
        font_weight="bold",
        font_color="#222222",
        ax=ax
    )

    # ---------------------------
    # 5. 辅助：统计每对节点有多少条边
    # ---------------------------
    pair_to_edges = defaultdict(list)
    for u, v, k in G.edges(keys=True):
        pair_to_edges[(u, v)].append(k)

    def get_rad(index, total):
        """同一对节点的多条边分开一些"""
        if total == 1:
            return 0.0
        step = 0.16
        center = (total - 1) / 2.0
        return (index - center) * step

    # ---------------------------
    # 6. 计算“曲线边上的标签位置”
    # ---------------------------
    def get_bezier_mid_and_angle(x1, y1, x2, y2, rad=0.0, t=0.5):
        """
        对于 arc3,rad=... 近似成二次贝塞尔曲线，
        返回曲线中点位置和该点切线角度。
        """
        # 中点
        mx = (x1 + x2) / 2.0
        my = (y1 + y2) / 2.0

        dx = x2 - x1
        dy = y2 - y1
        dist = math.hypot(dx, dy)
        if dist == 0:
            dist = 1.0

        # 法向量
        nxv = -dy / dist
        nyv = dx / dist

        # 控制点：沿法向偏移
        cx = mx + nxv * rad * dist
        cy = my + nyv * rad * dist

        # 二次贝塞尔点
        bx = (1 - t) ** 2 * x1 + 2 * (1 - t) * t * cx + t ** 2 * x2
        by = (1 - t) ** 2 * y1 + 2 * (1 - t) * t * cy + t ** 2 * y2

        # 二次贝塞尔导数（切线）
        tx = 2 * (1 - t) * (cx - x1) + 2 * t * (x2 - cx)
        ty = 2 * (1 - t) * (cy - y1) + 2 * t * (y2 - cy)

        angle = math.degrees(math.atan2(ty, tx))
        if angle > 90:
            angle -= 180
        if angle < -90:
            angle += 180

        return bx, by, angle

    # ---------------------------
    # 7. 逐条画边 + 在边上画标签
    # ---------------------------
    for u, v, k, data in G.edges(keys=True, data=True):
        relation = str(data.get("relation", "")).strip()

        keys = pair_to_edges[(u, v)]
        idx = keys.index(k)
        total = len(keys)

        if u == v:
            # 自环
            rad = 0.45 + idx * 0.08
            conn_style = f"arc3,rad={rad}"

            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=[(u, v, k)],
                arrowstyle="-|>",
                arrowsize=18,
                width=1.8,
                edge_color="#2E2E2E",
                connectionstyle=conn_style,
                min_source_margin=20,
                min_target_margin=20,
                ax=ax
            )

            x, y = pos[u]
            ax.text(
                x,
                y + 0.55 + idx * 0.12,
                relation,
                fontsize=10,
                ha="center",
                va="center",
                color="#222222",
                bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.15)
            )
        else:
            rad = get_rad(idx, total)

            # 如果存在反向边，再稍微分开一点
            if G.number_of_edges(v, u) > 0:
                rad += 0.08 if str(u) < str(v) else -0.08

            conn_style = f"arc3,rad={rad}"

            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=[(u, v, k)],
                arrowstyle="-|>",
                arrowsize=18,
                width=1.8,
                edge_color="#2E2E2E",
                connectionstyle=conn_style,
                min_source_margin=20,
                min_target_margin=20,
                ax=ax
            )

            x1, y1 = pos[u]
            x2, y2 = pos[v]

            # 标签直接算在曲线边上
            lx, ly, angle = get_bezier_mid_and_angle(x1, y1, x2, y2, rad=rad, t=0.52)

            ax.text(
                lx,
                ly,
                relation,
                fontsize=9.5,
                ha="center",
                va="center",
                rotation=angle,
                rotation_mode="anchor",
                color="#222222",
                bbox=dict(
                    fc="white",
                    ec="none",
                    alpha=0.68,
                    pad=0.10
                )
            )

    # ---------------------------
    # 8. 美化画布
    # ---------------------------
    ax.set_xlim(-3.0, 3.0)
    ax.set_ylim(-2.4, 2.6)
    ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_png, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
# =========================
# Convert to frontend graph
# =========================

def pre_to_frontend_graph(pre_data):
    objects = []
    relations = []

    number = pre_data.get("number", {})
    relationship = pre_data.get("relationship", [])

    for idx, obj_name in enumerate(number.keys()):
        x = 20 + (idx * 17) % 65
        y = 25 + (idx * 23) % 55

        objects.append(
            {
                "id": str(obj_name),
                "label": str(obj_name),
                "x": x,
                "y": y,
                "box": [
                    max(2, x - 6),
                    max(2, y - 5),
                    min(98, x + 8),
                    min(98, y + 7),
                ],
            }
        )

    object_set = {o["id"] for o in objects}

    for rel in relationship:
        if not isinstance(rel, (list, tuple)) or len(rel) != 3:
            continue

        subj, pred, obj = map(str, rel)

        if subj not in object_set:
            idx = len(objects)
            x = 20 + (idx * 17) % 65
            y = 25 + (idx * 23) % 55

            objects.append(
                {
                    "id": subj,
                    "label": subj,
                    "x": x,
                    "y": y,
                    "box": [
                        max(2, x - 6),
                        max(2, y - 5),
                        min(98, x + 8),
                        min(98, y + 7),
                    ],
                }
            )
            object_set.add(subj)

        if obj not in object_set:
            idx = len(objects)
            x = 20 + (idx * 17) % 65
            y = 25 + (idx * 23) % 55

            objects.append(
                {
                    "id": obj,
                    "label": obj,
                    "x": x,
                    "y": y,
                    "box": [
                        max(2, x - 6),
                        max(2, y - 5),
                        min(98, x + 8),
                        min(98, y + 7),
                    ],
                }
            )
            object_set.add(obj)

        relations.append(
            {
                "source": subj,
                "predicate": pred,
                "target": obj,
            }
        )

    return {
        "objects": objects,
        "relations": relations,
    }


# =========================
# API
# =========================

@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "STAR-SGG backend is running. Visit /docs for API docs.",
    }


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "sgg_dir": str(SGG_DIR),
        "eval_results": str(EVAL_RESULTS_PATH),
        "test_image_dir": str(TEST_IMAGE_DIR),
        "outputs": str(OUTPUT_DIR),
    }


@app.post("/api/sgg")
async def get_existing_sgg_by_uploaded_filename(image: UploadFile = File(...)):
    suffix = Path(image.filename).suffix.lower()

    if suffix not in [".jpg", ".jpeg", ".png"]:
        return {
            "status": "error",
            "message": "Only jpg, jpeg, png are supported.",
        }

    save_path = UPLOAD_DIR / image.filename

    with open(save_path, "wb") as f:
        shutil.copyfileobj(image.file, f)

    try:
        image_id = extract_image_id_from_filename(image.filename)
    except Exception:
        return {
            "status": "error",
            "message": (
                f"Cannot extract image_id from filename: {image.filename}. "
                "Expected format like 0004.png or 0004__xxx.png."
            ),
        }

    pkl_path = find_sgg_pickle(image_id)

    if pkl_path is None:
        return {
            "status": "error",
            "message": f"No SGG pickle found for image_id={image_id} in {SGG_DIR}",
            "image_id": image_id,
            "searched": [
                str(SGG_DIR / f"{image_id:04d}.pickle"),
                str(SGG_DIR / f"{image_id}.pickle"),
            ],
        }

    raw_data = load_pickle(pkl_path)
    pre_data = normalize_pre_data(raw_data, max_relations_for_demo=None)
    interactive_graph = build_interactive_graph(pre_data)

    # 1. 画 SGG 关系图
    graph_png = OUTPUT_DIR / f"{Path(image.filename).stem}_scene_graph.png"
    draw_scene_graph(pre_data, graph_png)

    # 2. 画原图检测框
    annotated_url = None
    prediction_index = None
    detection_error = None

    try:
        pred, prediction_index = get_prediction_by_filename(image.filename)

        annotated_png = OUTPUT_DIR / f"{Path(image.filename).stem}_annotated.png"
        draw_detection_boxes_on_image(
            image_path=save_path,
            pred=pred,
            output_path=annotated_png,
            score_thresh=0.3,
            max_boxes=80,
        )

        annotated_url = f"/outputs/{annotated_png.name}"

    except Exception as e:
        detection_error = repr(e)
        print(f"[WARN] failed to draw detection boxes for {image.filename}: {detection_error}")

    graph = pre_to_frontend_graph(pre_data)

    return {
        "status": "ok",
        "mode": "read_existing_pickle_and_eval_results",
        "image": image.filename,
        "image_id": image_id,
        "prediction_index": prediction_index,
        "sgg_pickle": str(pkl_path),
        "sceneGraphFile": f"/outputs/{pkl_path.name}",
        "annotatedImageUrl": annotated_url,
        "graphImageUrl": f"/outputs/{graph_png.name}",
        "detection_error": detection_error,
        "rawRelationCount": pre_data.get("raw_relation_count", 0),
        "displayRelationCount": pre_data.get("display_relation_count", 0),
        "pre": pre_data,
        "graph": interactive_graph,
    }

def normalize_category_name(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def infer_sgg_question_type(question: str, fallback_type: str = "") -> str:
    q = (question or "").lower()

    if any(w in q for w in ["how many", "count", "number", "quantity", "amount"]):
        return "count"

    if any(w in q for w in ["where", "location", "located", "position", "spatial"]):
        return "location"

    if any(w in q for w in ["relationship", "relation", "between", "connect", "adjacent", "intersect", "parked", "over", "through"]):
        return "relationship"

    if any(w in q for w in ["category", "categories", "class", "classes", "object type", "object categories"]):
        return "category"

    return fallback_type or "category"


def extract_target_categories(question: str, categories):
    q = (question or "").lower()
    found = []

    for cat in categories:
        cat = str(cat)
        variants = {
            cat.lower(),
            cat.replace("_", " ").lower(),
            cat.replace("_", "-").lower(),
        }

        for v in variants:
            if v and v in q:
                found.append(cat)
                break

    return found


def unique_list(items):
    result = []
    seen = set()

    for item in items:
        key = str(item)
        if key not in seen:
            seen.add(key)
            result.append(item)

    return result


def unique_relationships(relationships, max_items=None):
    result = []
    seen = set()

    for rel in relationships:
        if not isinstance(rel, (list, tuple)) or len(rel) != 3:
            continue

        subj, pred, obj = map(str, rel)
        key = (subj, pred, obj)

        if key in seen:
            continue

        seen.add(key)
        result.append((subj, pred, obj))

        if max_items is not None and len(result) >= max_items:
            break

    return result


def answer_category_question(image_name: str, pre_data: dict) -> str:
    number_info = pre_data.get("number", {})
    categories = list(number_info.keys())

    if not categories:
        categories = pre_data.get("category", [])

    categories = [str(x) for x in categories]

    if not categories:
        return f"No category information was found for image {image_name}."

    return f"The image {image_name} contains the following scene graph object categories: {', '.join(categories)}."


def answer_count_question(image_name: str, question: str, pre_data: dict) -> str:
    number_info = pre_data.get("number", {})
    categories = list(number_info.keys())
    targets = extract_target_categories(question, categories)

    if not number_info:
        return f"No quantity information was found for image {image_name}."

    if targets:
        lines = []
        for cat in targets:
            lines.append(f"- {cat}: {int(number_info.get(cat, 0))}")
        return f"The image {image_name} has the following object counts:\n" + "\n".join(lines)

    lines = []
    for cat, count in number_info.items():
        lines.append(f"- {cat}: {int(count)}")

    return f"The image {image_name} has the following object counts per category:\n" + "\n".join(lines)


def answer_location_question(image_name: str, question: str, pre_data: dict) -> str:
    location_info = pre_data.get("location", {})
    number_info = pre_data.get("number", {})
    categories = list(number_info.keys()) or list(location_info.keys())
    targets = extract_target_categories(question, categories)

    if not location_info:
        return f"No location information was found for image {image_name}."

    def format_locations(cat):
        locs = location_info.get(cat, [])
        if not isinstance(locs, list):
            locs = [locs]

        locs = [str(x) for x in locs]
        unique_locs = unique_list(locs)

        if not unique_locs:
            return f"- {cat}: no location information"

        if len(unique_locs) > 12:
            shown = ", ".join(unique_locs[:12])
            return f"- {cat}: {shown}, ... total {len(locs)} location records"

        return f"- {cat}: {', '.join(unique_locs)}"

    if targets:
        lines = [format_locations(cat) for cat in targets]
        return f"The location information in image {image_name} is:\n" + "\n".join(lines)

    lines = []
    for cat in categories:
        if cat in location_info:
            lines.append(format_locations(cat))

    if not lines:
        return f"No matching location information was found for image {image_name}."

    return f"The image {image_name} has the following object location information:\n" + "\n".join(lines)


def answer_relationship_question(image_name: str, question: str, pre_data: dict) -> str:
    relationships = pre_data.get("relationship", [])
    number_info = pre_data.get("number", {})

    categories = list(number_info.keys())
    if not categories:
        for rel in relationships:
            if isinstance(rel, (list, tuple)) and len(rel) == 3:
                categories.extend([str(rel[0]), str(rel[2])])
        categories = unique_list(categories)

    targets = extract_target_categories(question, categories)

    if not relationships:
        return f"No relationship information was found for image {image_name}."

    all_unique = unique_relationships(relationships)

    filtered = []

    if len(targets) >= 2:
        target_set = set(targets)
        for subj, pred, obj in all_unique:
            if subj in target_set and obj in target_set:
                filtered.append((subj, pred, obj))
    elif len(targets) == 1:
        target = targets[0]
        for subj, pred, obj in all_unique:
            if subj == target or obj == target:
                filtered.append((subj, pred, obj))
    else:
        filtered = all_unique

    if not filtered:
        return f"No matching relationships were found for image {image_name} based on the question."

    max_show = 40
    shown = filtered[:max_show]

    lines = []
    for idx, (subj, pred, obj) in enumerate(shown, 1):
        lines.append(f"{idx}. {subj} -- {pred} --> {obj}")

    answer = f"The image {image_name} has the following scene graph relationships:\n" + "\n".join(lines)

    if len(filtered) > max_show:
        answer += f"\n... {len(filtered) - max_show} more relationships are omitted."

    return answer


def answer_sgg_question_from_pre_data(image_name: str, question: str, question_type: str, pre_data: dict) -> str:
    inferred_type = infer_sgg_question_type(question, question_type)

    if inferred_type == "category":
        return answer_category_question(image_name, pre_data)

    if inferred_type == "count":
        return answer_count_question(image_name, question, pre_data)

    if inferred_type == "location":
        return answer_location_question(image_name, question, pre_data)

    if inferred_type == "relationship":
        return answer_relationship_question(image_name, question, pre_data)

    return answer_category_question(image_name, pre_data)

from pydantic import BaseModel


class SggQaRequest(BaseModel):
    type: str = ""
    question: str = ""
    image: str = ""


@app.post("/api/sgg-qa")
async def sgg_qa(req: SggQaRequest):
    image_name = req.image or ""
    question = req.question or ""
    question_type = req.type or ""

    if not image_name:
        return {
            "status": "error",
            "answer": "No image name was provided."
        }

    try:
        image_id = extract_image_id_from_filename(image_name)
    except Exception:
        return {
            "status": "error",
            "answer": f"Failed to extract image id from image name: {image_name}"
        }

    try:
        sgg_pickle = find_sgg_pickle(image_id)

        if not sgg_pickle or not sgg_pickle.exists():
            return {
                "status": "error",
                "answer": f"No SGG file was found for image {image_name}."
            }

        raw_data = load_pickle(sgg_pickle)
        pre_data = normalize_pre_data(raw_data, max_relations_for_demo=None)

        answer = answer_sgg_question_from_pre_data(
            image_name=image_name,
            question=question,
            question_type=question_type,
            pre_data=pre_data,
        )

        return {
            "status": "ok",
            "image": image_name,
            "image_id": image_id,
            "question_type": infer_sgg_question_type(question, question_type),
            "question": question,
            "answer": answer,
        }

    except Exception as e:
        return {
            "status": "error",
            "answer": f"SGG question answering failed: {repr(e)}"
        }


import re


def strip_options_from_question_text(text: str) -> str:
    """
    只保留问题文本，去掉 A/B/C/D/E... 选项。
    """
    return re.sub(
        r"\n\s*[A-Z]\.\s*[\s\S]*$",
        "",
        str(text or ""),
        flags=re.IGNORECASE,
    ).strip()


def extract_choice_letter(raw_answer: str) -> str:
    """
    从模型输出中提取选项字母。
    支持:
    B
    E
    RULE_SCORE:E
    Answer: C
    """
    s = str(raw_answer or "").strip()

    m = re.search(r"RULE_SCORE\s*:\s*([A-Z])", s, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    m = re.search(r"\b([A-Z])\b", s, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    return ""


def parse_options_from_question(question_text: str) -> dict:
    """
    从完整多选题里提取所有选项句子。
    支持 A-Z，不只支持 A-D。
    """
    text = str(question_text or "")

    pattern = re.compile(
        r"(?ms)^\s*([A-Z])\.\s*(.*?)(?=^\s*[A-Z]\.\s*|\Z)"
    )

    options = {}

    for m in pattern.finditer(text):
        letter = m.group(1).upper()
        content = m.group(2).strip()

        content = re.sub(
            r"\s*Answer with the option.*$",
            "",
            content,
            flags=re.IGNORECASE | re.DOTALL,
        ).strip()

        options[letter] = content

    return options


def normalize_question_for_match(text: str) -> str:
    """
    匹配时只比较问题文本，不比较选项。
    """
    text = strip_options_from_question_text(text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text

@app.post("/api/kg-rag")
async def kg_rag_answer(payload: dict):
    question_id = payload.get("question_id", None)
    image_name = payload.get("image", "")
    image_id = payload.get("image_id", None)
    input_question = payload.get("question", "")

    if image_id is None and image_name:
        try:
            image_id = extract_image_id_from_filename(image_name)
        except Exception:
            image_id = None

    result = None

    # 1. 优先 question_id 精确匹配
    if question_id is not None:
        result = find_first_kg_rag_result(
            image_id=image_id,
            question_id=question_id,
        )

    # 2. 没有 question_id，则根据 image_id + 纯问题文本匹配
    if result is None:
        results = load_kg_rag_results()

        candidates = []
        for item in results:
            if image_id is not None:
                try:
                    if int(item.get("image_id", -1)) != int(image_id):
                        continue
                except Exception:
                    continue

            candidates.append(item)

        input_norm = normalize_question_for_match(input_question)

        if input_norm:
            for item in candidates:
                q = str(item.get("question", ""))
                q_norm = normalize_question_for_match(q)

                if input_norm == q_norm or input_norm in q_norm or q_norm in input_norm:
                    result = item
                    break

        # 如果没有输入问题，或者没有匹配到，就取该图片第一条
        if result is None and candidates:
            result = candidates[0]

    if result is None:
        return {
            "status": "error",
            "answer": "No matched question was found in star_kg_rag_results2.json.",
            "question": input_question,
            "image": image_name,
            "image_id": image_id,
        }

    full_question = str(result.get("question", ""))
    question_only = strip_options_from_question_text(full_question)

    raw_llm_answer = result.get("llm_answer_raw", "")
    raw_ground_truth = result.get("ground_truth", "")

    choice_letter = extract_choice_letter(raw_llm_answer)
    gt_letter = extract_choice_letter(raw_ground_truth)

    options = parse_options_from_question(full_question)

    llm_answer_sentence = options.get(choice_letter, "")
    ground_truth_sentence = options.get(gt_letter, "")

    # 关键：answer 只返回选项对应句子，不返回 RULE_SCORE:E
    answer_sentence = llm_answer_sentence

    return {
        "status": "ok",
        "source_file": str(KG_RAG_RESULTS_PATH),
        "image": result.get("image", image_name),
        "image_id": result.get("image_id", image_id),
        "question_id": result.get("question_id"),

        "question": question_only,

        # 调试字段，可以保留，但前端不要显示
        "llm_answer_raw": choice_letter,
        "ground_truth": gt_letter,
        "ground_truth_sentence": ground_truth_sentence,
        "is_correct": result.get("is_correct", None),

        # 前端显示这个
        "llm_answer_sentence": answer_sentence,
        "answer": answer_sentence,

        "matched_result": result,
    }

@app.post("/api/common-vqa")
async def common_vqa(payload: dict):
    image_name = payload.get("image", "")
    question = payload.get("question", "")

    if not image_name:
        return {
            "status": "error",
            "answer": "I haven't received the name of the picture. Please upload the picture first.",
        }

    if not question:
        return {
            "status": "error",
            "answer": "I haven't received the question content. Please enter a question.",
        }

    try:
        image_id = extract_image_id_from_filename(image_name)
    except Exception:
        return {
            "status": "error",
            "answer": f"I couldn't parse the image_id from the image name {image_name}.",
        }

    pkl_path = find_sgg_pickle(image_id)

    if pkl_path is None:
        return {
            "status": "error",
            "answer": f"I couldn't find the SGG file for image {image_name}.",
        }

    raw_data = load_pickle(pkl_path)
    pre_data = normalize_pre_data(raw_data, max_relations_for_demo=None)
    sgg_text = sgg_to_text(pre_data, image_name=image_name)

    prompt = f"""
你是一个基于场景图生成（Scene Graph Generation, SGG）的遥感视觉问答助手。

你不能直接看到图像，只能根据下面提供的 SGG 信息回答问题。
SGG 信息包含目标类别、数量、位置以及目标之间的关系。
请严格依据 SGG 信息作答。
如果 SGG 信息不足，请回答：“根据当前 SGG 信息无法确定。”
不要编造任何未被 SGG 信息支持的细节。

【SGG 信息】
{sgg_text}

【用户问题】
{question}

请用中文回答。回答要简洁、清晰。
"""

    try:
        llm_answer = call_text_llm(prompt, max_new_tokens=256)
    except Exception as e:
        print(f"[LLM ERROR] {repr(e)}")
        return {
            "status": "error",
            "answer": f"Local LLM call failed: {repr(e)}",
            "prompt": prompt,
        }

    return {
        "status": "ok",
        "image": image_name,
        "image_id": image_id,
        "sgg_file": str(pkl_path),
        "question": question,
        "answer": llm_answer,
        "prompt": prompt,
    }
