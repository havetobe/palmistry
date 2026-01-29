const imageCanvas = document.getElementById("image-canvas");
const overlayCanvas = document.getElementById("overlay-canvas");
const emptyState = document.getElementById("empty-state");

const metricTime = document.getElementById("metric-time");
const metricHeart = document.getElementById("metric-heart");
const metricHead = document.getElementById("metric-head");
const metricLife = document.getElementById("metric-life");
const alerts = document.getElementById("alerts");
const analysisStatus = document.getElementById("analysis-status");
const analysisOverview = document.getElementById("analysis-overview");
const analysisTraits = document.getElementById("analysis-traits");
const analysisStrengths = document.getElementById("analysis-strengths");
const analysisChallenges = document.getElementById("analysis-challenges");
const analysisRelationships = document.getElementById("analysis-relationships");
const analysisCareers = document.getElementById("analysis-careers");
const analysisSelfcare = document.getElementById("analysis-selfcare");
const analysisConfidence = document.getElementById("analysis-confidence");
const analysisDisclaimer = document.getElementById("analysis-disclaimer");
const interpretButton = document.getElementById("btn-interpret");
const exportButton = document.getElementById("btn-export");
const metricHandedness = document.getElementById("metric-handedness");
const editToolbar = document.getElementById("edit-toolbar");
const cameraModal = document.getElementById("camera-modal");
const cameraVideo = document.getElementById("camera-video");
const cameraCanvas = document.getElementById("camera-canvas");
const cameraClose = document.getElementById("btn-camera-close");
const cameraCapture = document.getElementById("btn-camera-capture");
const cameraSwitch = document.getElementById("btn-camera-switch");

const state = {
  imageLoaded: false,
  lines: null,
  keypoints: null,
  confidences: null,
  warnings: [],
  timeMs: 0,
  roi: null,
  source: "demo",
  analysis: null,
  baseImageLoaded: false,
  editing: false,
  tool: "drag",
  activeLine: "heart",
  zoom: 1,
  originalLines: null,
  history: [],
  drag: null,
  show: {
    heart: true,
    head: true,
    life: true,
    keypoints: true,
  },
};

let cameraStream = null;
let cameraFacingMode = "environment";

const API_URL =
  window.PALMTRACE_API_URL || "http://127.0.0.1:8000/api/predict";
const ANALYSIS_URL =
  window.PALMTRACE_ANALYSIS_URL || "http://127.0.0.1:8000/api/palm/analysis";
const DEFAULT_ROI = { x: 0.12, y: 0.12, w: 0.76, h: 0.76 };

const colors = {
  heart: "#f06543",
  head: "#2b8fb3",
  life: "#efc45d",
  roi: "rgba(31, 26, 22, 0.15)",
  palmRoot: "#f06543",
  tigerMouth: "#2b8fb3",
  palmCenter: "#2fbf71",
};

const ctx = imageCanvas.getContext("2d");
const overlayCtx = overlayCanvas.getContext("2d");

function setCanvasSize(width, height) {
  imageCanvas.width = width;
  imageCanvas.height = height;
  overlayCanvas.width = width;
  overlayCanvas.height = height;
}

function applyCanvasTransform() {
  const scale = clamp(state.zoom, 1, 3);
  const origin = state.zoomOrigin || { x: 0.5, y: 0.5 };
  const originText = `${origin.x * 100}% ${origin.y * 100}%`;
  [imageCanvas, overlayCanvas].forEach((canvas) => {
    canvas.style.transformOrigin = originText;
    canvas.style.transform = scale === 1 ? "none" : `scale(${scale})`;
  });
}

function resetUI() {
  ctx.clearRect(0, 0, imageCanvas.width, imageCanvas.height);
  overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
  state.imageLoaded = false;
  state.lines = null;
  state.keypoints = null;
  state.confidences = null;
  state.warnings = [];
  state.timeMs = 0;
  state.roi = null;
  state.source = "demo";
  state.baseImageLoaded = false;
  state.editing = false;
  state.tool = "drag";
  state.activeLine = "heart";
  state.zoom = 1;
  state.zoomOrigin = { x: 0.5, y: 0.5 };
  state.originalLines = null;
  state.history = [];
  metricTime.textContent = "-";
  metricHandedness.textContent = "-";
  metricHeart.textContent = "-";
  metricHead.textContent = "-";
  metricLife.textContent = "-";
  alerts.innerHTML = "";
  state.analysis = null;
  interpretButton.disabled = true;
  exportButton.disabled = true;
  resetAnalysisUI();
  emptyState.style.display = "grid";
  applyCanvasTransform();
  updateToolUI();
}

async function openCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    document.getElementById("file-camera").click();
    return;
  }
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: cameraFacingMode },
      audio: false,
    });
    if (cameraVideo) {
      cameraVideo.srcObject = cameraStream;
    }
    if (cameraModal) {
      cameraModal.classList.add("show");
      cameraModal.setAttribute("aria-hidden", "false");
    }
  } catch (error) {
    document.getElementById("file-camera").click();
  }
}

function stopCamera() {
  if (cameraStream) {
    cameraStream.getTracks().forEach((track) => track.stop());
    cameraStream = null;
  }
  if (cameraVideo) {
    cameraVideo.srcObject = null;
  }
  if (cameraModal) {
    cameraModal.classList.remove("show");
    cameraModal.setAttribute("aria-hidden", "true");
  }
}

function captureFromCamera() {
  if (!cameraVideo || !cameraCanvas) {
    return;
  }
  const width = cameraVideo.videoWidth || 1280;
  const height = cameraVideo.videoHeight || 720;
  cameraCanvas.width = width;
  cameraCanvas.height = height;
  const cctx = cameraCanvas.getContext("2d");
  cctx.drawImage(cameraVideo, 0, 0, width, height);
  cameraCanvas.toBlob(
    (blob) => {
      if (!blob) {
        return;
      }
      const file = new File([blob], "camera.jpg", { type: "image/jpeg" });
      stopCamera();
      handleFile(file);
    },
    "image/jpeg",
    0.92
  );
}

function updateMetrics() {
  metricTime.textContent = state.timeMs ? `${state.timeMs.toFixed(0)} ms` : "-";
  if (metricHandedness) {
    const handedness = state.keypoints && state.keypoints.handedness;
    const label =
      handedness === "Left"
        ? "左手"
        : handedness === "Right"
        ? "右手"
        : "-";
    metricHandedness.textContent = label;
  }
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

function resetAnalysisUI() {
  analysisStatus.textContent = "尚未生成解读";
  analysisOverview.textContent = "-";
  analysisDisclaimer.textContent = "-";
  [analysisTraits, analysisStrengths, analysisChallenges, analysisRelationships, analysisCareers, analysisSelfcare, analysisConfidence].forEach(
    (list) => {
      list.innerHTML = "";
    }
  );
}

function renderList(target, items) {
  target.innerHTML = "";
  if (!items || !items.length) {
    const li = document.createElement("li");
    li.textContent = "-";
    target.appendChild(li);
    return;
  }
  items.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    target.appendChild(li);
  });
}

function renderAnalysis(analysis) {
  if (!analysis) {
    resetAnalysisUI();
    return;
  }
  analysisStatus.textContent = "解读已生成";
  analysisOverview.textContent = analysis.overview || "-";
  renderList(analysisTraits, analysis.personalityTraits || []);
  renderList(analysisStrengths, analysis.strengths || []);
  renderList(analysisChallenges, analysis.challenges || []);
  renderList(analysisRelationships, analysis.relationshipStyle || []);
  renderList(analysisCareers, analysis.careerInclinations || []);
  renderList(analysisSelfcare, analysis.selfCareTips || []);
  renderList(analysisConfidence, analysis.confidenceNotes || []);
  analysisDisclaimer.textContent =
    analysis.disclaimer || "仅供娱乐参考，不构成任何建议。";
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

function getScale() {
  const minSide = Math.min(overlayCanvas.width, overlayCanvas.height);
  return {
    lineWidth: clamp(minSide * 0.006, 2, 10),
    dashSize: clamp(minSide * 0.015, 6, 18),
    dashGap: clamp(minSide * 0.012, 4, 14),
    pointRadius: clamp(minSide * 0.01, 6, 18),
    crossSize: clamp(minSide * 0.015, 8, 22),
    fontSize: clamp(minSide * 0.015, 10, 16),
  };
}

function drawPolyline(points, color, dashed) {
  const absolutePoints = denormalize(points);
  const scale = getScale();
  overlayCtx.save();
  overlayCtx.strokeStyle = color;
  overlayCtx.lineWidth = scale.lineWidth;
  overlayCtx.lineJoin = "round";
  overlayCtx.lineCap = "round";
  if (dashed) {
    overlayCtx.setLineDash([scale.dashSize, scale.dashGap]);
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

function drawHandles(points, color) {
  const scale = getScale();
  const radius = state.editing
    ? Math.max(4, scale.pointRadius * 0.9)
    : Math.max(2, scale.pointRadius * 0.4);
  overlayCtx.save();
  overlayCtx.fillStyle = color;
  points.forEach((pt) => {
    const x = pt.x * overlayCanvas.width;
    const y = pt.y * overlayCanvas.height;
    overlayCtx.beginPath();
    overlayCtx.arc(x, y, radius, 0, Math.PI * 2);
    overlayCtx.fill();
  });
  overlayCtx.restore();
}

function cloneLines(lines) {
  if (!lines) {
    return null;
  }
  return {
    heart: (lines.heart || []).map((pt) => ({ x: pt.x, y: pt.y })),
    head: (lines.head || []).map((pt) => ({ x: pt.x, y: pt.y })),
    life: (lines.life || []).map((pt) => ({ x: pt.x, y: pt.y })),
  };
}

function pushHistory() {
  if (!state.lines) {
    return;
  }
  state.history.push(cloneLines(state.lines));
  if (state.history.length > 20) {
    state.history.shift();
  }
}

function drawKeypoints(keypoints) {
  if (!keypoints) {
    return;
  }
  const width = overlayCanvas.width;
  const height = overlayCanvas.height;
  const flipped = Boolean(keypoints.flipped);
  const scale = getScale();
  const pointRadius = scale.pointRadius;
  const crossSize = scale.crossSize;
  const items = [
    { key: "palm_root", label: "掌根", color: colors.palmRoot },
    { key: "tiger_mouth", label: "虎口", color: colors.tigerMouth },
    { key: "palm_center", label: "掌心", color: colors.palmCenter },
  ];

  overlayCtx.save();
  overlayCtx.lineWidth = Math.max(2, pointRadius / 4);
  overlayCtx.font = `${scale.fontSize}px serif`;
  items.forEach((item) => {
    const pt = keypoints[item.key];
    if (!pt) {
      return;
    }
    let x = pt.x * width;
    const y = pt.y * height;
    if (flipped) {
      x = width - x;
    }
    overlayCtx.fillStyle = item.color;
    overlayCtx.strokeStyle = item.color;
    overlayCtx.beginPath();
    overlayCtx.arc(x, y, pointRadius, 0, Math.PI * 2);
    overlayCtx.fill();
    overlayCtx.beginPath();
    overlayCtx.moveTo(x - crossSize, y);
    overlayCtx.lineTo(x + crossSize, y);
    overlayCtx.moveTo(x, y - crossSize);
    overlayCtx.lineTo(x, y + crossSize);
    overlayCtx.stroke();
    overlayCtx.fillText(item.label, x + 10, y - 8);
  });
  overlayCtx.restore();
}

function drawOverlay() {
  overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
  if (!state.lines) {
    return;
  }
  if (state.source === "backend" && !state.baseImageLoaded) {
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
    const pts = mapToRoi(state.lines.heart, roi);
    drawPolyline(pts, colors.heart, false);
    if (state.editing) {
      drawHandles(pts, colors.heart);
    }
  }
  if (state.show.head) {
    const pts = mapToRoi(state.lines.head, roi);
    drawPolyline(pts, colors.head, true);
    if (state.editing) {
      drawHandles(pts, colors.head);
    }
  }
  if (state.show.life) {
    const pts = mapToRoi(state.lines.life, roi);
    drawPolyline(pts, colors.life, false);
    if (state.editing) {
      drawHandles(pts, colors.life);
    }
  }
  if (state.show.keypoints) {
    drawKeypoints(state.keypoints);
  }
}

function getPointerPosition(event) {
  const rect = overlayCanvas.getBoundingClientRect();
  const cssX = event.clientX - rect.left;
  const cssY = event.clientY - rect.top;
  const scale = clamp(state.zoom, 1, 3);
  const baseCssW = rect.width / scale;
  const baseCssH = rect.height / scale;
  const origin = state.zoomOrigin || { x: 0.5, y: 0.5 };
  const ox = baseCssW * origin.x;
  const oy = baseCssH * origin.y;
  const oxScaled = ox * scale;
  const oyScaled = oy * scale;
  const unscaledCssX = ox + (cssX - oxScaled) / scale;
  const unscaledCssY = oy + (cssY - oyScaled) / scale;
  const x = unscaledCssX * (overlayCanvas.width / baseCssW);
  const y = unscaledCssY * (overlayCanvas.height / baseCssH);
  return { x, y };
}

function findNearestPoint(px, py) {
  if (!state.lines) {
    return null;
  }
  const roi = state.roi || DEFAULT_ROI;
  const scale = getScale();
  const threshold = state.editing
    ? Math.max(14, scale.pointRadius * 1.8)
    : Math.max(8, scale.pointRadius * 1.2);
  const candidates = [];
  const lineOrder = [
    { key: "heart", enabled: state.show.heart },
    { key: "head", enabled: state.show.head },
    { key: "life", enabled: state.show.life },
  ];
  lineOrder.forEach((line) => {
    if (!line.enabled || !state.lines[line.key]) {
      return;
    }
    const pts = mapToRoi(state.lines[line.key], roi);
    pts.forEach((pt, idx) => {
      const x = pt.x * overlayCanvas.width;
      const y = pt.y * overlayCanvas.height;
      const dx = x - px;
      const dy = y - py;
      const dist = Math.hypot(dx, dy);
      if (dist <= threshold) {
        candidates.push({ key: line.key, idx, dist });
      }
    });
  });
  if (!candidates.length) {
    return null;
  }
  candidates.sort((a, b) => a.dist - b.dist);
  return candidates[0];
}

function updatePointFromPointer(lineKey, idx, px, py) {
  const roi = state.roi || DEFAULT_ROI;
  const nx = px / overlayCanvas.width;
  const ny = py / overlayCanvas.height;
  const x = clamp((nx - roi.x) / roi.w, 0, 1);
  const y = clamp((ny - roi.y) / roi.h, 0, 1);
  const line = state.lines && state.lines[lineKey];
  if (!line || !line[idx]) {
    return;
  }
  line[idx] = { x, y };
}

function insertPoint(lineKey, px, py) {
  const roi = state.roi || DEFAULT_ROI;
  const line = state.lines && state.lines[lineKey];
  if (!line || line.length < 2) {
    return;
  }
  const nx = px / overlayCanvas.width;
  const ny = py / overlayCanvas.height;
  const x = clamp((nx - roi.x) / roi.w, 0, 1);
  const y = clamp((ny - roi.y) / roi.h, 0, 1);
  const pts = mapToRoi(line, roi);
  let bestIndex = 0;
  let bestDist = Infinity;
  for (let i = 0; i < pts.length - 1; i += 1) {
    const ax = pts[i].x * overlayCanvas.width;
    const ay = pts[i].y * overlayCanvas.height;
    const bx = pts[i + 1].x * overlayCanvas.width;
    const by = pts[i + 1].y * overlayCanvas.height;
    const dx = bx - ax;
    const dy = by - ay;
    const len2 = dx * dx + dy * dy || 1;
    const t = Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / len2));
    const projX = ax + t * dx;
    const projY = ay + t * dy;
    const dist = Math.hypot(px - projX, py - projY);
    if (dist < bestDist) {
      bestDist = dist;
      bestIndex = i + 1;
    }
  }
  line.splice(bestIndex, 0, { x, y });
}

function updateToolUI() {
  if (!editToolbar) {
    return;
  }
  editToolbar.classList.toggle("show", state.editing);
  editToolbar.querySelectorAll(".tool-btn").forEach((btn) => {
    if (btn.dataset.tool) {
      btn.classList.toggle("active", btn.dataset.tool === state.tool);
    }
    if (btn.dataset.line) {
      btn.classList.toggle("active", btn.dataset.line === state.activeLine);
    }
  });
  if (!state.editing) {
    overlayCanvas.style.cursor = "default";
    return;
  }
  overlayCanvas.style.cursor = state.tool === "erase" ? "not-allowed" : "grab";
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

function handleFile(file) {
  if (!file) {
    return;
  }
  const reader = new FileReader();
  reader.onload = () => {
    const img = new Image();
    img.onload = () => {
      setCanvasSize(img.naturalWidth, img.naturalHeight);
      ctx.clearRect(0, 0, imageCanvas.width, imageCanvas.height);
      ctx.drawImage(img, 0, 0);
      applyCanvasTransform();
      state.imageLoaded = true;
      emptyState.style.display = "none";
      requestBackend(file).catch((error) => {
        state.lines = null;
        state.confidences = null;
        state.timeMs = (error && error.timeMs) || 0;
        const baseMessage =
          (error && error.message) ||
          "Backend unavailable. Start api_server.py and retry.";
        const suggestions = (error && error.suggestions) || [];
        state.warnings = [baseMessage, ...suggestions];
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
  state.originalLines = cloneLines(state.lines);
  state.history = [];
  state.keypoints = data.keypoints || null;
  state.confidences = data.confidences || null;
  state.timeMs = data.time_ms || 0;
  state.warnings = data.warnings || [];
  state.roi = data.roi || DEFAULT_ROI;
  state.source = "backend";
  state.analysis = data.analysis || null;
  state.baseImageLoaded = false;
  if (data.base_image) {
    renderBackendImage(data.base_image);
  } else {
    state.baseImageLoaded = true;
  }
  updateToolUI();
  interpretButton.disabled = !state.lines;
  exportButton.disabled = !state.lines;
  renderAnalysis(state.analysis);
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
    state.baseImageLoaded = true;
    applyCanvasTransform();
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
      suggestions: data.suggestions || [],
      timeMs: data.time_ms,
    };
  }
  applyBackendResult(data);
}

async function requestAnalysis() {
  if (!state.lines) {
    analysisStatus.textContent = "请先完成掌纹识别";
    return;
  }
  analysisStatus.textContent = "正在生成解读…";
  const payload = {
    lines: state.lines,
    confidences: state.confidences,
    roi: state.roi || DEFAULT_ROI,
  };
  const response = await fetch(ANALYSIS_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await response.json();
  if (!response.ok || !data.ok) {
    analysisStatus.textContent = data.error || "解读生成失败";
    return;
  }
  state.analysis = data.analysis || null;
  renderAnalysis(state.analysis);
}

document.getElementById("btn-camera").addEventListener("click", () => {
  openCamera();
});

document.getElementById("btn-upload").addEventListener("click", () => {
  document.getElementById("file-upload").click();
});

document.getElementById("btn-reset").addEventListener("click", () => {
  resetUI();
});

exportButton.addEventListener("click", () => {
  if (!state.lines) {
    return;
  }
  const payload = {
    version: 1,
    generated_at: new Date().toISOString(),
    roi: state.roi || DEFAULT_ROI,
    lines: state.lines,
    confidences: state.confidences,
    keypoints: state.keypoints,
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "palm_result.json";
  a.click();
  URL.revokeObjectURL(url);
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

document.getElementById("toggle-keypoints").addEventListener("change", (event) => {
  state.show.keypoints = event.target.checked;
  drawOverlay();
});

if (cameraClose) {
  cameraClose.addEventListener("click", () => {
    stopCamera();
  });
}

if (cameraCapture) {
  cameraCapture.addEventListener("click", () => {
    captureFromCamera();
  });
}

if (cameraSwitch) {
  cameraSwitch.addEventListener("click", async () => {
    cameraFacingMode = cameraFacingMode === "environment" ? "user" : "environment";
    stopCamera();
    await openCamera();
  });
}

document.getElementById("toggle-edit").addEventListener("change", (event) => {
  state.editing = event.target.checked;
  state.drag = null;
  overlayCanvas.style.touchAction = state.editing ? "none" : "auto";
  overlayCanvas.style.cursor = state.editing ? "grab" : "default";
  state.zoom = state.editing ? state.zoom : 1;
  state.zoomOrigin = { x: 0.5, y: 0.5 };
  updateToolUI();
  applyCanvasTransform();
  drawOverlay();
});

if (editToolbar) {
  editToolbar.addEventListener("click", (event) => {
    const button = event.target.closest(".tool-btn");
    if (!button) {
      return;
    }
    if (button.dataset.line) {
      state.activeLine = button.dataset.line;
      updateToolUI();
      return;
    }
    const tool = button.dataset.tool;
    if (!tool) {
      return;
    }
    if (tool === "undo") {
      const previous = state.history.pop();
      if (previous) {
        state.lines = previous;
      }
      drawOverlay();
      return;
    }
    if (tool === "reset-line") {
      if (state.originalLines && state.lines) {
        pushHistory();
        state.lines[state.activeLine] = (state.originalLines[state.activeLine] || []).map(
          (pt) => ({ x: pt.x, y: pt.y })
        );
        drawOverlay();
      }
      return;
    }
    if (tool === "zoom-in") {
      state.zoom = clamp(state.zoom + 0.2, 1, 3);
      applyCanvasTransform();
      return;
    }
    if (tool === "zoom-out") {
      state.zoom = clamp(state.zoom - 0.2, 1, 3);
      applyCanvasTransform();
      return;
    }
    state.tool = tool;
    updateToolUI();
  });
}

overlayCanvas.addEventListener("pointerdown", (event) => {
  if (!state.editing) {
    return;
  }
  const { x, y } = getPointerPosition(event);
  if (state.tool === "add") {
    if (!state.lines) {
      return;
    }
    pushHistory();
    insertPoint(state.activeLine, x, y);
    drawOverlay();
    return;
  }
  const hit = findNearestPoint(x, y);
  if (!hit) {
    return;
  }
  if (state.tool === "erase") {
    const line = state.lines && state.lines[hit.key];
    if (line && line.length > 2) {
      pushHistory();
      line.splice(hit.idx, 1);
    }
    drawOverlay();
    return;
  }
  overlayCanvas.setPointerCapture(event.pointerId);
  pushHistory();
  state.drag = { key: hit.key, idx: hit.idx };
  updatePointFromPointer(hit.key, hit.idx, x, y);
  drawOverlay();
});

overlayCanvas.addEventListener("pointermove", (event) => {
  if (!state.editing || !state.drag) {
    return;
  }
  const { x, y } = getPointerPosition(event);
  updatePointFromPointer(state.drag.key, state.drag.idx, x, y);
  drawOverlay();
});

overlayCanvas.addEventListener("pointerup", (event) => {
  if (state.drag) {
    overlayCanvas.releasePointerCapture(event.pointerId);
  }
  state.drag = null;
});

overlayCanvas.addEventListener("pointerleave", () => {
  state.drag = null;
});

overlayCanvas.addEventListener("wheel", (event) => {
  if (!state.editing) {
    return;
  }
  event.preventDefault();
  const rect = overlayCanvas.getBoundingClientRect();
  const ox = (event.clientX - rect.left) / rect.width;
  const oy = (event.clientY - rect.top) / rect.height;
  state.zoomOrigin = { x: clamp(ox, 0, 1), y: clamp(oy, 0, 1) };
  const delta = event.deltaY > 0 ? -0.15 : 0.15;
  state.zoom = clamp(state.zoom + delta, 1, 3);
  applyCanvasTransform();
});

interpretButton.addEventListener("click", () => {
  requestAnalysis().catch(() => {
    analysisStatus.textContent = "解读请求失败，请稍后重试";
  });
});

resetUI();
