const imageCanvas = document.getElementById("image-canvas");
const overlayCanvas = document.getElementById("overlay-canvas");
const emptyState = document.getElementById("empty-state");

const metricTime = document.getElementById("metric-time");
const metricHeart = document.getElementById("metric-heart");
const metricHead = document.getElementById("metric-head");
const metricLife = document.getElementById("metric-life");
const alerts = document.getElementById("alerts");

const state = {
  imageLoaded: false,
  lines: null,
  confidences: null,
  warnings: [],
  timeMs: 0,
  roi: null,
  source: "demo",
  show: {
    heart: true,
    head: true,
    life: true,
  },
};

const API_URL =
  window.PALMTRACE_API_URL || "http://127.0.0.1:8000/api/predict";
const DEFAULT_ROI = { x: 0.12, y: 0.12, w: 0.76, h: 0.76 };

const colors = {
  heart: "#f06543",
  head: "#2b8fb3",
  life: "#efc45d",
  roi: "rgba(31, 26, 22, 0.15)",
};

const ctx = imageCanvas.getContext("2d");
const overlayCtx = overlayCanvas.getContext("2d");

function setCanvasSize(width, height) {
  imageCanvas.width = width;
  imageCanvas.height = height;
  overlayCanvas.width = width;
  overlayCanvas.height = height;
}

function resetUI() {
  ctx.clearRect(0, 0, imageCanvas.width, imageCanvas.height);
  overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
  state.imageLoaded = false;
  state.lines = null;
  state.confidences = null;
  state.warnings = [];
  state.timeMs = 0;
  state.roi = null;
  state.source = "demo";
  metricTime.textContent = "-";
  metricHeart.textContent = "-";
  metricHead.textContent = "-";
  metricLife.textContent = "-";
  alerts.innerHTML = "";
  emptyState.style.display = "grid";
}

function updateMetrics() {
  metricTime.textContent = state.timeMs ? `${state.timeMs.toFixed(0)} ms` : "-";
  metricHeart.textContent = state.confidences
    ? `${Math.round(state.confidences.heart * 100)}%`
    : "-";
  metricHead.textContent = state.confidences
    ? `${Math.round(state.confidences.head * 100)}%`
    : "-";
  metricLife.textContent = state.confidences
    ? `${Math.round(state.confidences.life * 100)}%`
    : "-";
}

function updateAlerts(messages) {
  alerts.innerHTML = "";
  if (!messages.length) {
    const info = document.createElement("div");
    info.className = "alert info";
    info.textContent =
      state.source === "backend"
        ? "模型识别成功，线条已叠加显示。"
        : "演示线条已生成（未连接模型服务）。";
    alerts.appendChild(info);
    return;
  }

  messages.forEach((msg) => {
    const item = document.createElement("div");
    item.className = "alert warn";
    item.textContent = msg;
    alerts.appendChild(item);
  });
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function computeBrightness() {
  const sample = 32;
  const width = imageCanvas.width;
  const height = imageCanvas.height;
  const stepX = Math.max(1, Math.floor(width / sample));
  const stepY = Math.max(1, Math.floor(height / sample));
  const data = ctx.getImageData(0, 0, width, height).data;
  let total = 0;
  let count = 0;

  for (let y = 0; y < height; y += stepY) {
    for (let x = 0; x < width; x += stepX) {
      const idx = (y * width + x) * 4;
      const r = data[idx];
      const g = data[idx + 1];
      const b = data[idx + 2];
      total += 0.2126 * r + 0.7152 * g + 0.0722 * b;
      count += 1;
    }
  }

  return total / count;
}

function generateLines() {
  const lines = {};
  const points = 20;

  lines.heart = Array.from({ length: points }, (_, i) => {
    const t = i / (points - 1);
    const x = 0.12 + 0.76 * t;
    const y = 0.28 - 0.05 * Math.sin(Math.PI * t) + 0.02 * t;
    return { x, y };
  });

  lines.head = Array.from({ length: points }, (_, i) => {
    const t = i / (points - 1);
    const x = 0.18 + 0.72 * t;
    const y = 0.46 + 0.03 * Math.sin(Math.PI * 1.2 * t);
    return { x, y };
  });

  lines.life = Array.from({ length: points }, (_, i) => {
    const t = i / (points - 1);
    const angle = Math.PI * (0.15 + 0.8 * t);
    const x = 0.3 + 0.2 * Math.cos(angle);
    const y = 0.18 + 0.62 * Math.sin(angle);
    return { x, y };
  });

  return lines;
}

function mapToRoi(points, roi) {
  if (!roi) {
    return points;
  }
  return points.map((pt) => ({
    x: roi.x + pt.x * roi.w,
    y: roi.y + pt.y * roi.h,
  }));
}

function denormalize(points) {
  return points.map((pt) => [pt.x * overlayCanvas.width, pt.y * overlayCanvas.height]);
}

function drawPolyline(points, color, dashed) {
  const absolutePoints = denormalize(points);
  overlayCtx.save();
  overlayCtx.strokeStyle = color;
  overlayCtx.lineWidth = 4;
  overlayCtx.lineJoin = "round";
  overlayCtx.lineCap = "round";
  if (dashed) {
    overlayCtx.setLineDash([10, 8]);
  } else {
    overlayCtx.setLineDash([]);
  }
  overlayCtx.beginPath();
  absolutePoints.forEach(([x, y], index) => {
    if (index === 0) {
      overlayCtx.moveTo(x, y);
    } else {
      overlayCtx.lineTo(x, y);
    }
  });
  overlayCtx.stroke();
  overlayCtx.restore();
}

function drawOverlay() {
  overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
  if (!state.lines) {
    return;
  }

  const roi = state.roi || DEFAULT_ROI;
  overlayCtx.save();
  overlayCtx.strokeStyle = colors.roi;
  overlayCtx.lineWidth = 2;
  overlayCtx.strokeRect(
    overlayCanvas.width * roi.x,
    overlayCanvas.height * roi.y,
    overlayCanvas.width * roi.w,
    overlayCanvas.height * roi.h
  );
  overlayCtx.restore();

  if (state.show.heart) {
    drawPolyline(mapToRoi(state.lines.heart, roi), colors.heart, false);
  }
  if (state.show.head) {
    drawPolyline(mapToRoi(state.lines.head, roi), colors.head, true);
  }
  if (state.show.life) {
    drawPolyline(mapToRoi(state.lines.life, roi), colors.life, false);
  }
}

function analyzeImage(fallbackReason) {
  const start = performance.now();
  const warnings = [];
  const brightness = computeBrightness();
  const minSide = Math.min(imageCanvas.width, imageCanvas.height);

  if (fallbackReason) {
    warnings.push(fallbackReason);
  }
  if (brightness < 60) {
    warnings.push("光照过暗，请在明亮环境重新拍摄。");
  }
  if (minSide < 720) {
    warnings.push("分辨率偏低，识别效果可能受影响。");
  }

  state.lines = null;
  state.confidences = null;
  state.timeMs = performance.now() - start;
  state.warnings = warnings;
  state.source = "demo";
  drawOverlay();
  updateMetrics();
  updateAlerts(warnings);
}

function getExifOrientation(buffer) {
  const view = new DataView(buffer);
  if (view.getUint16(0, false) !== 0xffd8) {
    return 1;
  }
  let offset = 2;
  const length = view.byteLength;
  while (offset < length) {
    const marker = view.getUint16(offset, false);
    offset += 2;
    if (marker === 0xffe1) {
      const exifLength = view.getUint16(offset, false);
      offset += 2;
      if (view.getUint32(offset, false) !== 0x45786966) {
        break;
      }
      offset += 6;
      const little = view.getUint16(offset, false) === 0x4949;
      offset += view.getUint32(offset + 4, little);
      const tags = view.getUint16(offset, little);
      offset += 2;
      for (let i = 0; i < tags; i += 1) {
        const tagOffset = offset + i * 12;
        if (view.getUint16(tagOffset, little) === 0x0112) {
          return view.getUint16(tagOffset + 8, little);
        }
      }
      break;
    } else if ((marker & 0xff00) !== 0xff00) {
      break;
    } else {
      offset += view.getUint16(offset, false);
    }
  }
  return 1;
}

function drawImageWithOrientation(img, orientation) {
  const width = img.naturalWidth;
  const height = img.naturalHeight;
  const rotate = [5, 6, 7, 8].includes(orientation);
  setCanvasSize(rotate ? height : width, rotate ? width : height);

  ctx.save();
  switch (orientation) {
    case 2:
      ctx.translate(width, 0);
      ctx.scale(-1, 1);
      break;
    case 3:
      ctx.translate(width, height);
      ctx.rotate(Math.PI);
      break;
    case 4:
      ctx.translate(0, height);
      ctx.scale(1, -1);
      break;
    case 5:
      ctx.rotate(0.5 * Math.PI);
      ctx.scale(1, -1);
      break;
    case 6:
      ctx.rotate(0.5 * Math.PI);
      ctx.translate(0, -height);
      break;
    case 7:
      ctx.rotate(0.5 * Math.PI);
      ctx.translate(width, -height);
      ctx.scale(-1, 1);
      break;
    case 8:
      ctx.rotate(-0.5 * Math.PI);
      ctx.translate(-width, 0);
      break;
    default:
      break;
  }
  ctx.drawImage(img, 0, 0);
  ctx.restore();
}

function handleFile(file) {
  if (!file) {
    return;
  }
  const reader = new FileReader();
  reader.onload = () => {
    const buffer = reader.result;
    const orientation = getExifOrientation(buffer);
    const img = new Image();
    img.onload = () => {
      drawImageWithOrientation(img, orientation);
      state.imageLoaded = true;
      emptyState.style.display = "none";
      requestBackend(file).catch((error) => {
        state.lines = null;
        state.confidences = null;
        state.timeMs = (error && error.timeMs) || 0;
        state.warnings = [
          (error && error.message) ||
            "Backend unavailable. Start api_server.py and retry.",
        ];
        state.source = "backend";
        drawOverlay();
        updateMetrics();
        updateAlerts(state.warnings);
      });
      URL.revokeObjectURL(img.src);
    };
    img.src = URL.createObjectURL(file);
  };
  reader.readAsArrayBuffer(file);
}

function applyBackendResult(data) {
  state.lines = data.lines || null;
  state.confidences = data.confidences || null;
  state.timeMs = data.time_ms || 0;
  state.warnings = data.warnings || [];
  state.roi = data.roi || DEFAULT_ROI;
  state.source = "backend";
  if (data.base_image) {
    renderBackendImage(data.base_image);
  }
  drawOverlay();
  updateMetrics();
  updateAlerts(state.warnings);
}

function renderBackendImage(dataUrl) {
  const img = new Image();
  img.onload = () => {
    setCanvasSize(img.naturalWidth, img.naturalHeight);
    ctx.clearRect(0, 0, imageCanvas.width, imageCanvas.height);
    ctx.drawImage(img, 0, 0);
    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    // Repaint overlays after backend image replaces the base canvas.
    drawOverlay();
  };
  img.src = dataUrl;
}

async function requestBackend(file) {
  const form = new FormData();
  form.append("image", file);
  const response = await fetch(API_URL, {
    method: "POST",
    body: form,
  });
  const data = await response.json();
  if (!response.ok || !data.ok) {
    throw {
      kind: "backend",
      message: data.error || "模型识别失败。",
      timeMs: data.time_ms,
    };
  }
  applyBackendResult(data);
}

document.getElementById("btn-camera").addEventListener("click", () => {
  document.getElementById("file-camera").click();
});

document.getElementById("btn-upload").addEventListener("click", () => {
  document.getElementById("file-upload").click();
});

document.getElementById("btn-reset").addEventListener("click", () => {
  resetUI();
});

document.getElementById("file-camera").addEventListener("change", (event) => {
  handleFile(event.target.files[0]);
  event.target.value = "";
});

document.getElementById("file-upload").addEventListener("change", (event) => {
  handleFile(event.target.files[0]);
  event.target.value = "";
});

document.getElementById("toggle-heart").addEventListener("change", (event) => {
  state.show.heart = event.target.checked;
  drawOverlay();
});

document.getElementById("toggle-head").addEventListener("change", (event) => {
  state.show.head = event.target.checked;
  drawOverlay();
});

document.getElementById("toggle-life").addEventListener("change", (event) => {
  state.show.life = event.target.checked;
  drawOverlay();
});

resetUI();
