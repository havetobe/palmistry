import base64
import io
import json
import math
import os
import time
import uuid
import urllib.error
import urllib.request
from typing import Dict, List, Tuple

from flask import Flask, jsonify, request
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import torch

from tools import remove_background, resize, extract_semantic_keypoints
from model import UNet
from rectification import warp, warp_with_matrix
from detection import detect
from classification import classify

APP = Flask(__name__)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(ROOT_DIR, "input")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
MODEL_PATH = os.path.join(ROOT_DIR, "checkpoint", "checkpoint_aug_epoch70.pth")
PROJECT_DIR = os.path.dirname(ROOT_DIR)

RESIZE_VALUE = 256
LINES_WIDTH = RESIZE_VALUE
LINES_HEIGHT = RESIZE_VALUE

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def read_env_file(path: str) -> Dict[str, str]:
    if not os.path.exists(path):
        return {}
    values: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                values[key] = value
    return values


def read_config_file(path: str) -> Dict[str, str]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    except json.JSONDecodeError:
        return {}
    return {}


def load_runtime_config() -> Dict[str, str]:
    settings: Dict[str, str] = {}
    for config_path in (
        os.path.join(PROJECT_DIR, "config.json"),
        os.path.join(ROOT_DIR, "config.json"),
    ):
        settings.update(read_config_file(config_path))
    for env_path in (
        os.path.join(PROJECT_DIR, ".env"),
        os.path.join(ROOT_DIR, ".env"),
    ):
        settings.update(read_env_file(env_path))
    for key, value in os.environ.items():
        settings[key] = value
    return settings


RUNTIME_CONFIG = load_runtime_config()


def get_setting(key: str, default: str = "") -> str:
    value = RUNTIME_CONFIG.get(key)
    if value is None or str(value).strip() == "":
        return default
    return str(value)


def load_model(device: torch.device) -> UNet:
    net = UNet(n_channels=3, n_classes=1).to(device)
    net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    net.eval()
    return net


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = load_model(DEVICE)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def line_confidence(line: List[List[int]]) -> float:
    length = len(line)
    # Boost confidence slightly to avoid overly low percentages for typical lengths.
    raw = length / 120.0
    scaled = 0.55 + 0.4 * clamp(raw, 0.0, 1.0)
    return clamp(scaled, 0.55, 0.95)


def normalize_line(
    line: List[List[int]], width: int, height: int, stride: int = 4
) -> List[Dict[str, float]]:
    points = []
    for i, (y, x, _, _) in enumerate(line):
        if i % stride != 0 and i != len(line) - 1:
            continue
        points.append({"x": x / width, "y": y / height})
    return points


def line_length(points: List[Dict[str, float]]) -> float:
    if len(points) < 2:
        return 0.0
    total = 0.0
    for idx in range(1, len(points)):
        dx = points[idx]["x"] - points[idx - 1]["x"]
        dy = points[idx]["y"] - points[idx - 1]["y"]
        total += math.hypot(dx, dy)
    return total


def line_curvature(points: List[Dict[str, float]]) -> float:
    if len(points) < 3:
        return 0.0
    total = 0.0
    samples = 0
    for idx in range(1, len(points) - 1):
        x0, y0 = points[idx - 1]["x"], points[idx - 1]["y"]
        x1, y1 = points[idx]["x"], points[idx]["y"]
        x2, y2 = points[idx + 1]["x"], points[idx + 1]["y"]
        v1x, v1y = x1 - x0, y1 - y0
        v2x, v2y = x2 - x1, y2 - y1
        norm1 = math.hypot(v1x, v1y)
        norm2 = math.hypot(v2x, v2y)
        if norm1 == 0 or norm2 == 0:
            continue
        cos_angle = clamp((v1x * v2x + v1y * v2y) / (norm1 * norm2), -1.0, 1.0)
        total += math.acos(cos_angle)
        samples += 1
    return total / max(1, samples)


def line_span(points: List[Dict[str, float]]) -> Dict[str, float]:
    if not points:
        return {"min_x": 0.0, "max_x": 0.0, "min_y": 0.0, "max_y": 0.0}
    xs = [p["x"] for p in points]
    ys = [p["y"] for p in points]
    return {
        "min_x": min(xs),
        "max_x": max(xs),
        "min_y": min(ys),
        "max_y": max(ys),
    }


def summarize_line(points: List[Dict[str, float]]) -> Dict[str, float]:
    if not points:
        return {}
    start = points[0]
    end = points[-1]
    span = line_span(points)
    return {
        "point_count": len(points),
        "length": round(line_length(points), 4),
        "curvature": round(line_curvature(points), 4),
        "start_x": round(start["x"], 4),
        "start_y": round(start["y"], 4),
        "end_x": round(end["x"], 4),
        "end_y": round(end["y"], 4),
        "span_x": round(span["max_x"] - span["min_x"], 4),
        "span_y": round(span["max_y"] - span["min_y"], 4),
    }


def build_palm_summary(
    lines: Dict[str, List[Dict[str, float]]],
    confidences: Dict[str, float],
    roi: Dict[str, float],
) -> Dict[str, object]:
    summary = {
        "version": 1,
        "lines": {},
        "confidences": confidences or {},
        "roi": roi or {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0},
    }
    for key, points in (lines or {}).items():
        summary["lines"][key] = summarize_line(points)
    return summary


def build_palm_prompt(summary: Dict[str, object]) -> str:
    return (
        "你是手相解读与性格倾向分析助手（仅供娱乐）。"
        "请基于掌纹特征（心线/智慧线/生命线）生成友好、非决定论的解读。"
        "避免医学/法律/财务建议，不要断言事实。"
        "输出严格JSON（不要Markdown）。必须字段："
        "overview（字符串），personalityTraits（字符串数组），strengths（字符串数组），"
        "challenges（字符串数组），relationshipStyle（字符串数组），"
        "careerInclinations（字符串数组），selfCareTips（字符串数组），"
        "confidenceNotes（字符串数组），disclaimer（字符串）。"
        "请用中文撰写所有输出内容。"
        "输入数据："
        + json.dumps(summary, ensure_ascii=True)
    )


def extract_json_block(content: str) -> str:
    if not content:
        return ""
    trimmed = content.strip()
    if trimmed.startswith("```"):
        start = trimmed.find("{")
        end = trimmed.rfind("}")
        if start >= 0 and end > start:
            return trimmed[start : end + 1]
    if trimmed.startswith("{") and trimmed.endswith("}"):
        return trimmed
    start = trimmed.find("{")
    end = trimmed.rfind("}")
    if start >= 0 and end > start:
        return trimmed[start : end + 1]
    return trimmed


def parse_assistant_content(response_text: str) -> Dict[str, object]:
    if not response_text:
        raise ValueError("empty assistant response")
    content = response_text
    try:
        payload = json.loads(response_text)
        if isinstance(payload, dict) and "choices" in payload:
            choices = payload.get("choices") or []
            if choices:
                message = choices[0].get("message") or {}
                content_obj = message.get("content")
                if isinstance(content_obj, str):
                    content = content_obj
                elif isinstance(content_obj, list):
                    parts = []
                    for item in content_obj:
                        if isinstance(item, dict) and item.get("type") == "text":
                            parts.append(item.get("text", ""))
                    content = "".join(parts)
    except json.JSONDecodeError:
        content = response_text
    json_text = extract_json_block(content)
    return json.loads(json_text)


def call_palm_assistant(summary: Dict[str, object], user_id: str) -> Dict[str, object]:
    endpoint = get_setting(
        "PALM_ASSISTANT_ENDPOINT",
        "https://yuanqi.tencent.com/openapi/v1/agent/chat/completions",
    )
    api_key = get_setting("PALM_ASSISTANT_API_KEY", "")
    assistant_id = get_setting("PALM_ASSISTANT_ID", "")
    if not api_key or not assistant_id:
        return {
            "status": "skipped",
            "reason": "assistant_not_configured",
            "disclaimer": "Palm reading output is for entertainment only.",
        }

    body = {
        "assistant_id": assistant_id.strip(),
        "user_id": f"palm-{user_id or uuid.uuid4().hex}",
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": build_palm_prompt(summary)}],
            }
        ],
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key.strip()}",
    }
    request_data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(endpoint, data=request_data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            if resp.status < 200 or resp.status >= 300:
                raise RuntimeError(f"assistant call failed: {resp.status}")
            response_text = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"assistant call failed: {exc.code} {detail}") from exc
    return parse_assistant_content(response_text)


def _parse_pad_ratio(raw_value: str, default: float = 0.12) -> float:
    if raw_value is None:
        return default
    try:
        value = float(str(raw_value).strip())
    except (TypeError, ValueError):
        return default
    return clamp(value, 0.0, 0.4)


def preprocess_image(file_bytes: bytes, pad_ratio: float = 0.12) -> Dict[str, object]:
    image = Image.open(io.BytesIO(file_bytes))
    image = ImageOps.exif_transpose(image).convert("RGB")
    orig_w, orig_h = image.size
    original_filename = f"upload_original_{uuid.uuid4().hex}.png"
    original_path = os.path.join(INPUT_DIR, original_filename)
    image.save(original_path, "PNG")
    pad = int(min(image.size) * pad_ratio)
    if pad > 0:
        image = ImageOps.expand(image, border=pad, fill=(12, 12, 12))
    filename = f"upload_{uuid.uuid4().hex}.png"
    input_path = os.path.join(INPUT_DIR, filename)
    image.save(input_path, "PNG")
    return {
        "path": input_path,
        "original_path": original_path,
        "orig_size": (orig_w, orig_h),
        "padded_size": image.size,
        "pad": pad,
    }


def _run_once(
    file_bytes: bytes, pad_ratio: float
) -> Tuple[
    Dict[str, List[Dict[str, float]]],
    Dict[str, float],
    str,
    Dict[str, object],
    str,
    str,
    Tuple[int, int],
]:
    preprocess = preprocess_image(file_bytes, pad_ratio=pad_ratio)
    input_path = preprocess["path"]
    original_path = preprocess["original_path"]
    orig_size = preprocess["orig_size"]
    pad = preprocess["pad"]
    padded_size = preprocess["padded_size"]
    base_image_data_url = load_image_data_url(original_path)
    try:
        lines, confidences, base_image, keypoints, error_code = run_pipeline(
            input_path, orig_size=orig_size, pad=pad, padded_size=padded_size
        )
        if lines:
            save_result_image(
                original_path,
                lines,
                keypoints,
                os.path.join(RESULTS_DIR, "result.jpg"),
            )
        return (
            lines,
            confidences,
            base_image_data_url,
            keypoints,
            error_code,
            original_path,
            orig_size,
        )
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)
        if os.path.exists(original_path):
            os.remove(original_path)


def load_image_data_url(path: str) -> str:
    if not path or not os.path.exists(path):
        return ""
    _, ext = os.path.splitext(path)
    ext = (ext or "").lower()
    mime = "image/png" if ext == ".png" else "image/jpeg"
    with open(path, "rb") as handle:
        encoded = base64.b64encode(handle.read()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def map_points_to_original(
    points: List[Dict[str, float]],
    homography: np.ndarray,
    warped_size: Tuple[int, int],
    orig_size: Tuple[int, int],
    pad: int = 0,
) -> List[Dict[str, float]]:
    if not points:
        return points
    warped_w, warped_h = warped_size
    orig_w, orig_h = orig_size
    inv = np.linalg.inv(homography)
    mapped = []
    for pt in points:
        xw = pt["x"] * warped_w
        yw = pt["y"] * warped_h
        vec = np.array([xw, yw, 1.0], dtype=np.float64)
        src = inv @ vec
        if src[2] == 0:
            continue
        xf = src[0] / src[2]
        yf = src[1] / src[2]
        x_unflip = warped_w - xf
        y_unflip = yf
        x_unpadded = x_unflip - pad
        y_unpadded = y_unflip - pad
        x_norm = clamp(x_unpadded / max(1, orig_w), 0.0, 1.0)
        y_norm = clamp(y_unpadded / max(1, orig_h), 0.0, 1.0)
        mapped.append({"x": float(x_norm), "y": float(y_norm)})
    return mapped


def extend_polyline_normalized(
    points: List[Dict[str, float]],
    orig_size: Tuple[int, int],
    ratio: float = 0.2,
    min_px: float = 12.0,
    max_px: float = 60.0,
) -> List[Dict[str, float]]:
    if len(points) < 2:
        return points
    orig_w, orig_h = orig_size
    p0 = points[-2]
    p1 = points[-1]
    dx = (p1["x"] - p0["x"]) * orig_w
    dy = (p1["y"] - p0["y"]) * orig_h
    seg_len = math.hypot(dx, dy)
    if seg_len == 0:
        return points
    total_len = 0.0
    for idx in range(1, len(points)):
        vx = (points[idx]["x"] - points[idx - 1]["x"]) * orig_w
        vy = (points[idx]["y"] - points[idx - 1]["y"]) * orig_h
        total_len += math.hypot(vx, vy)
    extend_len = clamp(total_len * ratio, min_px, max_px)
    ux = dx / seg_len
    uy = dy / seg_len
    new_x = clamp(p1["x"] + (extend_len * ux) / max(1, orig_w), 0.0, 1.0)
    new_y = clamp(p1["y"] + (extend_len * uy) / max(1, orig_h), 0.0, 1.0)
    return points + [{"x": float(new_x), "y": float(new_y)}]


def run_pipeline(
    input_path: str,
    orig_size: Tuple[int, int] = None,
    pad: int = 0,
    padded_size: Tuple[int, int] = None,
) -> Tuple[
    Dict[str, List[Dict[str, float]]],
    Dict[str, float],
    str,
    Dict[str, object],
    str,
]:
    clean_path = os.path.join(RESULTS_DIR, "palm_without_background.jpg")
    warped_path = os.path.join(RESULTS_DIR, "warped_palm.jpg")
    warped_clean_path = os.path.join(RESULTS_DIR, "warped_palm_clean.jpg")
    warped_mini_path = os.path.join(RESULTS_DIR, "warped_palm_mini.jpg")
    warped_clean_mini_path = os.path.join(RESULTS_DIR, "warped_palm_clean_mini.jpg")
    palmline_path = os.path.join(RESULTS_DIR, "palm_lines.png")

    remove_background(input_path, clean_path)

    fallback_used = False
    warp_result, homography, warped_size = warp_with_matrix(input_path, warped_path)
    if warp_result is None:
        fallback_used = True
        homography = None
        warped_size = padded_size or orig_size
        warped_path = input_path

    remove_background(warped_path, warped_clean_path)
    resize(
        warped_path,
        warped_clean_path,
        warped_mini_path,
        warped_clean_mini_path,
        RESIZE_VALUE,
    )

    detect(MODEL, warped_clean_path, palmline_path, RESIZE_VALUE, device=DEVICE)
    keypoints_payload = build_keypoints_payload(input_path, orig_size=orig_size, pad=pad)
    lines = classify(palmline_path)
    if lines is None or len(lines) < 3 or any(line is None for line in lines):
        return {}, {}, "", {}, "line_detection_failed"

    with open(warped_mini_path, "rb") as result_file:
        result_b64 = base64.b64encode(result_file.read()).decode("ascii")
    result_data_url = f"data:image/jpeg;base64,{result_b64}"

    output_lines = {
        "heart": normalize_line(lines[0], LINES_WIDTH, LINES_HEIGHT),
        "head": normalize_line(lines[1], LINES_WIDTH, LINES_HEIGHT),
        "life": normalize_line(lines[2], LINES_WIDTH, LINES_HEIGHT),
    }
    if homography is not None and orig_size:
        mapped = {
            key: map_points_to_original(value, homography, warped_size, orig_size, pad)
            for key, value in output_lines.items()
        }
        if mapped.get("life"):
            mapped["life"] = extend_polyline_normalized(mapped["life"], orig_size)
        output_lines = mapped
    elif orig_size and padded_size:
        padded_w, padded_h = padded_size
        orig_w, orig_h = orig_size
        mapped = {}
        for key, value in output_lines.items():
            mapped_points = []
            for pt in value:
                x_padded = pt["x"] * padded_w
                y_padded = pt["y"] * padded_h
                x_unpadded = x_padded - pad
                y_unpadded = y_padded - pad
                x_norm = clamp(x_unpadded / max(1, orig_w), 0.0, 1.0)
                y_norm = clamp(y_unpadded / max(1, orig_h), 0.0, 1.0)
                mapped_points.append({"x": float(x_norm), "y": float(y_norm)})
            mapped[key] = mapped_points
        output_lines = mapped
    confidences = {
        "heart": line_confidence(lines[0]),
        "head": line_confidence(lines[1]),
        "life": line_confidence(lines[2]),
    }
    if fallback_used:
        return output_lines, confidences, result_data_url, keypoints_payload, "warp_fallback"
    return output_lines, confidences, result_data_url, keypoints_payload, ""


def build_failure_payload(error_code: str) -> Tuple[str, List[str]]:
    suggestions = []
    if error_code == "warp_failed":
        message = "未检测到完整手掌，请调整拍摄角度"
        suggestions = [
            "保证手掌完整出现在画面内（不要被边缘裁切）",
            "掌心正对镜头，避免明显倾斜或旋转",
            "光线均匀、背景简洁",
        ]
    elif error_code == "line_detection_failed":
        message = "掌纹识别失败，请尝试更清晰的手掌照片"
        suggestions = [
            "提高分辨率或靠近拍摄",
            "避免过曝或过暗，尽量均匀光照",
            "掌心放平、纹理清晰可见",
        ]
    else:
        message = "识别失败，请重试"
        suggestions = [
            "确保手掌完整清晰、光线充足",
            "避免手指遮挡掌心区域",
        ]
    return message, suggestions


def save_result_image(
    original_path: str,
    lines: Dict[str, List[Dict[str, float]]],
    keypoints: Dict[str, object],
    output_path: str,
) -> None:
    if not original_path or not lines:
        return
    im = Image.open(original_path).convert("RGB")
    draw = ImageDraw.Draw(im)
    width = 6
    colors = {
        "heart": (255, 0, 0),
        "head": (0, 128, 0),
        "life": (0, 0, 255),
    }
    img_w, img_h = im.size
    for key in ("heart", "head", "life"):
        pts = lines.get(key) or []
        if len(pts) < 2:
            continue
        pixel_pts = [(p["x"] * img_w, p["y"] * img_h) for p in pts]
        draw.line(pixel_pts, fill=colors[key], width=width)
    if keypoints:
        point_radius = 16
        kp_colors = {
            "palm_root": (255, 69, 0),
            "tiger_mouth": (0, 191, 255),
            "palm_center": (0, 200, 0),
        }
        for key in ("palm_root", "tiger_mouth", "palm_center"):
            pt = keypoints.get(key)
            if not isinstance(pt, dict):
                continue
            x = pt.get("x")
            y = pt.get("y")
            if x is None or y is None:
                continue
            cx = x * img_w
            cy = y * img_h
            color = kp_colors.get(key, (255, 69, 0))
            draw.ellipse(
                (cx - point_radius, cy - point_radius, cx + point_radius, cy + point_radius),
                fill=color,
                outline=color,
                width=3,
            )
            draw.line(
                (cx - point_radius, cy, cx + point_radius, cy),
                fill=color,
                width=3,
            )
            draw.line(
                (cx, cy - point_radius, cx, cy + point_radius),
                fill=color,
                width=3,
            )
    im.save(output_path)


def build_keypoints_payload(
    path_to_image: str, orig_size: Tuple[int, int] = None, pad: int = 0
) -> Dict[str, object]:
    keypoints = extract_semantic_keypoints(path_to_image)
    if not keypoints:
        return {}
    image_width, image_height = keypoints.get("image_size", [0, 0])
    if not image_width or not image_height:
        return {}
    target_w, target_h = orig_size if orig_size else (image_width, image_height)

    def normalize_point(point: List[float]) -> Dict[str, float]:
        x, y = point
        if orig_size:
            x -= pad
            y -= pad
            return {
                "x": clamp(x / max(1, target_w), 0.0, 1.0),
                "y": clamp(y / max(1, target_h), 0.0, 1.0),
            }
        return {"x": x / image_width, "y": y / image_height}

    return {
        "palm_root": normalize_point(keypoints["palm_root"]),
        "tiger_mouth": normalize_point(keypoints["tiger_mouth"]),
        "palm_center": normalize_point(keypoints["palm_center"]),
        "flipped": bool(keypoints.get("flipped")),
        "handedness": keypoints.get("handedness", ""),
    }


def run_palm_analysis(
    lines: Dict[str, List[Dict[str, float]]],
    confidences: Dict[str, float],
    roi: Dict[str, float],
    user_id: str = "",
) -> Dict[str, object]:
    summary = build_palm_summary(lines, confidences, roi)
    try:
        analysis = call_palm_assistant(summary, user_id)
        return {"status": "ok", "analysis": analysis, "summary": summary}
    except Exception as exc:
        APP.logger.exception("Palm assistant failed")
        return {
            "status": "error",
            "error": str(exc),
            "summary": summary,
            "disclaimer": "Palm reading output is for entertainment only.",
        }


@APP.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    return response


@APP.route("/api/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return ("", 204)

    if "image" not in request.files:
        message, suggestions = build_failure_payload("warp_failed")
        return (
            jsonify({"ok": False, "error": message, "suggestions": suggestions}),
            400,
        )

    file = request.files["image"]
    if not file or file.filename == "":
        message, suggestions = build_failure_payload("warp_failed")
        return (
            jsonify({"ok": False, "error": message, "suggestions": suggestions}),
            400,
        )

    start = time.time()
    pad_ratio = _parse_pad_ratio(
        request.form.get("pad_ratio") or request.args.get("pad_ratio"), default=0.12
    )
    file_bytes = file.read()
    try:
        (
            lines,
            confidences,
            base_image_data_url,
            keypoints,
            error_code,
            _,
            _,
        ) = _run_once(file_bytes, pad_ratio)
        if pad_ratio == 0.0 and error_code == "warp_fallback":
            retry_pad = 0.06
            (
                retry_lines,
                retry_confidences,
                retry_base_image,
                retry_keypoints,
                retry_error_code,
                _,
                _,
            ) = _run_once(file_bytes, retry_pad)
            if retry_lines and retry_error_code == "":
                lines = retry_lines
                confidences = retry_confidences
                base_image_data_url = retry_base_image
                keypoints = retry_keypoints
                error_code = retry_error_code
    except Exception as exc:
        APP.logger.exception("Pipeline failed")
        return (
            jsonify({"ok": False, "error": "pipeline failed", "detail": str(exc)}),
            500,
        )

    if not lines:
        message, suggestions = build_failure_payload("line_detection_failed")
        return jsonify(
            {
                "ok": False,
                "error": message,
                "suggestions": suggestions,
                "time_ms": int((time.time() - start) * 1000),
            }
        )

    warnings = []
    if error_code == "warp_fallback":
        warnings.append("校正失败，已使用原图识别，结果可能略有偏差。")

    response = {
        "ok": True,
        "lines": lines,
        "confidences": confidences,
        "time_ms": int((time.time() - start) * 1000),
        "warnings": warnings,
        "roi": {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0},
        "base_image": base_image_data_url,
        "keypoints": keypoints,
    }
    with_analysis = (request.form.get("analysis") or request.args.get("analysis") or "").lower()
    if with_analysis in {"1", "true", "yes"}:
        analysis_result = run_palm_analysis(lines, confidences, response["roi"])
        response["analysis_status"] = analysis_result.get("status")
        response["analysis"] = analysis_result.get("analysis")
        response["analysis_error"] = analysis_result.get("error")
        response["analysis_summary"] = analysis_result.get("summary")
        response["disclaimer"] = analysis_result.get("disclaimer")
    return jsonify(response)


@APP.route("/api/palm/analysis", methods=["POST", "OPTIONS"])
def palm_analysis():
    if request.method == "OPTIONS":
        return ("", 204)

    start = time.time()
    user_id = request.form.get("user_id") or request.args.get("user_id") or ""
    lines: Dict[str, List[Dict[str, float]]] = {}
    confidences: Dict[str, float] = {}
    roi = {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}
    base_image = ""
    keypoints: Dict[str, object] = {}

    if "image" in request.files:
        file = request.files["image"]
        if not file or file.filename == "":
            message, suggestions = build_failure_payload("warp_failed")
            return (
                jsonify({"ok": False, "error": message, "suggestions": suggestions}),
                400,
            )
        pad_ratio = _parse_pad_ratio(
            request.form.get("pad_ratio") or request.args.get("pad_ratio"), default=0.12
        )
        file_bytes = file.read()
        try:
            (
                lines,
                confidences,
                base_image_data_url,
                keypoints,
                error_code,
                _,
                _,
            ) = _run_once(file_bytes, pad_ratio)
            if pad_ratio == 0.0 and error_code == "warp_fallback":
                retry_pad = 0.06
                (
                    retry_lines,
                    retry_confidences,
                    retry_base_image,
                    retry_keypoints,
                    retry_error_code,
                    _,
                    _,
                ) = _run_once(file_bytes, retry_pad)
                if retry_lines and retry_error_code == "":
                    lines = retry_lines
                    confidences = retry_confidences
                    base_image_data_url = retry_base_image
                    keypoints = retry_keypoints
                    error_code = retry_error_code
        except Exception as exc:
            APP.logger.exception("Pipeline failed")
            return (
                jsonify({"ok": False, "error": "pipeline failed", "detail": str(exc)}),
                500,
            )
        base_image_out = base_image_data_url
        warnings = []
        if error_code == "warp_fallback":
            warnings.append("校正失败，已使用原图识别，结果可能略有偏差。")
    else:
        payload = request.get_json(silent=True) or {}
        user_id = str(payload.get("user_id") or payload.get("userId") or "")
        lines = payload.get("lines") or {}
        confidences = payload.get("confidences") or {}
        roi = payload.get("roi") or roi
        base_image = payload.get("base_image") or ""
        base_image_out = base_image

    if not lines:
        message, suggestions = build_failure_payload("line_detection_failed")
        return jsonify(
            {
                "ok": False,
                "error": message,
                "suggestions": suggestions,
                "time_ms": int((time.time() - start) * 1000),
            }
        )

    analysis_result = run_palm_analysis(lines, confidences, roi, user_id=user_id)
    return jsonify(
        {
            "ok": analysis_result.get("status") == "ok",
            "analysis_status": analysis_result.get("status"),
            "analysis": analysis_result.get("analysis"),
            "analysis_error": analysis_result.get("error"),
            "analysis_summary": analysis_result.get("summary"),
            "disclaimer": analysis_result.get("disclaimer"),
            "lines": lines,
            "confidences": confidences,
            "roi": roi,
            "base_image": base_image_out,
            "keypoints": keypoints,
            "warnings": warnings if "warnings" in locals() else [],
            "time_ms": int((time.time() - start) * 1000),
        }
    )


@APP.route("/api/palm/analysis/receive", methods=["POST", "OPTIONS"])
def palm_analysis_receive():
    if request.method == "OPTIONS":
        return ("", 204)
    payload = request.get_json(silent=True) or {}
    user_id = payload.get("userId") or payload.get("user_id") or ""
    analysis_result = payload.get("analysisResult") or payload.get("analysis_result") or ""
    if not analysis_result:
        return jsonify({"ok": False, "error": "missing analysisResult"}), 400
    return jsonify({"ok": True, "user_id": user_id, "received": True})


if __name__ == "__main__":
    APP.run(host="0.0.0.0", port=8000, debug=False)
