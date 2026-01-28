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
from PIL import Image, ImageOps
import torch

from tools import remove_background, resize
from model import UNet
from rectification import warp
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


def preprocess_image(file_bytes: bytes) -> str:
    image = Image.open(io.BytesIO(file_bytes))
    image = ImageOps.exif_transpose(image).convert("RGB")
    filename = f"upload_{uuid.uuid4().hex}.png"
    input_path = os.path.join(INPUT_DIR, filename)
    image.save(input_path, "PNG")
    return input_path


def run_pipeline(
    input_path: str,
) -> Tuple[Dict[str, List[Dict[str, float]]], Dict[str, float], str]:
    clean_path = os.path.join(RESULTS_DIR, "palm_without_background.jpg")
    warped_path = os.path.join(RESULTS_DIR, "warped_palm.jpg")
    warped_clean_path = os.path.join(RESULTS_DIR, "warped_palm_clean.jpg")
    warped_mini_path = os.path.join(RESULTS_DIR, "warped_palm_mini.jpg")
    warped_clean_mini_path = os.path.join(RESULTS_DIR, "warped_palm_clean_mini.jpg")
    palmline_path = os.path.join(RESULTS_DIR, "palm_lines.png")

    remove_background(input_path, clean_path)

    warp_result = warp(input_path, warped_path)
    if warp_result is None:
        return {}, {}, ""

    remove_background(warped_path, warped_clean_path)
    resize(
        warped_path,
        warped_clean_path,
        warped_mini_path,
        warped_clean_mini_path,
        RESIZE_VALUE,
    )

    detect(MODEL, warped_clean_path, palmline_path, RESIZE_VALUE, device=DEVICE)
    lines = classify(palmline_path)
    if lines is None or len(lines) < 3 or any(line is None for line in lines):
        return {}, {}, ""

    with open(warped_mini_path, "rb") as result_file:
        result_b64 = base64.b64encode(result_file.read()).decode("ascii")
    result_data_url = f"data:image/jpeg;base64,{result_b64}"

    output_lines = {
        "heart": normalize_line(lines[0], LINES_WIDTH, LINES_HEIGHT),
        "head": normalize_line(lines[1], LINES_WIDTH, LINES_HEIGHT),
        "life": normalize_line(lines[2], LINES_WIDTH, LINES_HEIGHT),
    }
    confidences = {
        "heart": line_confidence(lines[0]),
        "head": line_confidence(lines[1]),
        "life": line_confidence(lines[2]),
    }
    return output_lines, confidences, result_data_url


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
        return jsonify({"ok": False, "error": "missing image file"}), 400

    file = request.files["image"]
    if not file or file.filename == "":
        return jsonify({"ok": False, "error": "empty filename"}), 400

    start = time.time()
    input_path = preprocess_image(file.read())

    try:
        lines, confidences, base_image = run_pipeline(input_path)
    except Exception as exc:
        APP.logger.exception("Pipeline failed")
        return (
            jsonify({"ok": False, "error": "pipeline failed", "detail": str(exc)}),
            500,
        )
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)

    if not lines:
        return jsonify(
            {
                "ok": False,
                "error": "no palm lines detected",
                "time_ms": int((time.time() - start) * 1000),
            }
        )

    response = {
        "ok": True,
        "lines": lines,
        "confidences": confidences,
        "time_ms": int((time.time() - start) * 1000),
        "warnings": [],
        "roi": {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0},
        "base_image": base_image,
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

    if "image" in request.files:
        file = request.files["image"]
        if not file or file.filename == "":
            return jsonify({"ok": False, "error": "empty filename"}), 400
        input_path = preprocess_image(file.read())
        try:
            lines, confidences, base_image = run_pipeline(input_path)
        except Exception as exc:
            APP.logger.exception("Pipeline failed")
            return (
                jsonify({"ok": False, "error": "pipeline failed", "detail": str(exc)}),
                500,
            )
        finally:
            if os.path.exists(input_path):
                os.remove(input_path)
    else:
        payload = request.get_json(silent=True) or {}
        user_id = str(payload.get("user_id") or payload.get("userId") or "")
        lines = payload.get("lines") or {}
        confidences = payload.get("confidences") or {}
        roi = payload.get("roi") or roi
        base_image = payload.get("base_image") or ""

    if not lines:
        return jsonify(
            {
                "ok": False,
                "error": "no palm lines provided",
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
            "base_image": base_image,
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
