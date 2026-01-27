# PalmTrace Debug Notes

Date: 2026-01-28  
Repo: `e:\model-1.26\Internship\CarryCode\palmistry`

## 1) H5 frontend refresh
- Issue: after upload, three lines did not redraw until toggles were changed.
- Fix: `mobile/h5/app.js` -> call `drawOverlay()` after backend image loads in `renderBackendImage()`.

## 2) Confidence always ~40%
- Cause: backend clamp used `clamp(raw, 0.4, 0.95)` so short lines floor to 40%.
- Change: `code/api_server.py` -> smoother mapping (0.2 + 0.7 * clamp(raw, 0..1), then clamp 0.05..0.95).

## 3) Line lengths short
- Added logging for line lengths in `code/api_server.py`.
- Found lengths like 40/35/30 => low confidence.

## 4) palm_lines.png black / no lines
- Cause: palm mask too small -> masked to zero.
- Fix: `code/tools.py` -> if palm_mask area < 2%, fall back to full mask.

## 5) skel_base=0 (classification stage)
- Found `gray stats: nonzero=0` even when image had values.
- Cause: classification re-applied palm_mask and wiped lines.
- Fix: `code/classification.py` -> if masked gray too small, keep original image (no re-mask).

## 6) Right-bottom noise -> life line misclassified
- Tried multiple suppressions (static region, dynamic distance).
- Current state: noise still present on some samples.

## 7) Three lines overlap (heart/head/life)
- Tried rule-based scoring with overlap penalties and y-separation.
- Implemented Scheme B (KMeans clustering on features).
- Added combo search with overlap penalties + y-order constraint.
- Still overlapped on many samples.

## 8) Structural fix: connected-component constraint
- New: label each line with skeleton connected-component id.
- If >= 3 components, force selecting lines from different components.
- Else, penalize choosing multiple lines from same component.
- File: `code/classification.py`

## 9) Detection threshold fallback
- `code/detection.py`: if thresholded mask too small, fallback to top-percentile threshold.
- Also fixed `detach()` for numpy conversion.

## Key logs to watch
- `filter_palm_lines pixels: raw/masked/cleaned`
- `gray stats: nonzero/min/max/binary_nonzero`
- `skel pixels: base/pruned`
- `lines: grouped / grouped_fallback / after_filter`
- `selected_lengths`

## Next steps
- If overlap persists, check if skeleton has only 1 component (root cause is segmentation output).
- Consider stronger noise filtering on `palm_lines.png` generation or revise segmentation model.
