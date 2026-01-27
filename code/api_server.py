import base64
import io
import os
import time
import uuid
from typing import Dict, List, Tuple

from flask import Flask, jsonify, request
from PIL import Image, ImageOps
import torch

from tools import remove_background, resize, filter_palm_lines
from model import UNet
from rectification import warp
from detection import detect
from classification import classify

APP = Flask(__name__)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(ROOT_DIR, "input")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
MODEL_PATH = os.path.join(ROOT_DIR, "checkpoint", "checkpoint_aug_epoch70.pth")

RESIZE_VALUE = 256
LINES_WIDTH = RESIZE_VALUE
LINES_HEIGHT = RESIZE_VALUE

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_model(device: torch.device) -> UNet:
    net = UNet(n_channels=3, n_classes=1).to(device)
    net.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    return net


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = load_model(DEVICE)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def line_confidence(line: List[List[int]]) -> float:
    length = len(line)
    # Map length to a smoother confidence range without a hard 40% floor.
    raw = length / 180.0
    scaled = 0.2 + 0.7 * clamp(raw, 0.0, 1.0)
    return clamp(scaled, 0.05, 0.95)


def normalize_line(
    line: List[List[int]], width: int, height: int, stride: int = 4
) -> List[Dict[str, float]]:
    points = []
    for i, (y, x, _, _) in enumerate(line):
        if i % stride != 0 and i != len(line) - 1:
            continue
        points.append({"x": x / width, "y": y / height})
    return points


def preprocess_image(file_bytes: bytes, original_name: str) -> str:
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
    resize(warped_path, warped_clean_path, warped_mini_path, warped_clean_mini_path, RESIZE_VALUE)

    detect(MODEL, warped_clean_path, palmline_path, RESIZE_VALUE, device=DEVICE)
    filter_palm_lines(palmline_path, warped_clean_mini_path)
    lines = classify(palmline_path, warped_clean_mini_path)
    if lines is None or len(lines) < 3 or any(line is None for line in lines):
        return {}, {}, ""
    print(
        "[palmtrace] line lengths:",
        "heart=",
        len(lines[0]),
        "head=",
        len(lines[1]),
        "life=",
        len(lines[2]),
    )

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
    input_path = preprocess_image(file.read(), file.filename)

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

    return jsonify(
        {
            "ok": True,
            "lines": lines,
            "confidences": confidences,
            "time_ms": int((time.time() - start) * 1000),
            "roi": {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0},
            "base_image": base_image,
        }
    )


if __name__ == "__main__":
    APP.run(host="0.0.0.0", port=8000, debug=False)
