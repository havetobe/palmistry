import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
from pillow_heif import register_heif_opener

def heic_to_jpeg(heic_dir, jpeg_dir):
    register_heif_opener()  
    image = Image.open(heic_dir)
    image.save(jpeg_dir, "JPEG")

def remove_background(jpeg_dir, path_to_clean_image):
    if jpeg_dir[-4:] in ['heic', 'HEIC']:
        heic_to_jpeg(jpeg_dir, jpeg_dir[:-4] + 'jpg')
        jpeg_dir = jpeg_dir[:-4] + 'jpg'
    img = cv2.imread(jpeg_dir)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 20, 80], dtype="uint8")
    upper = np.array([50, 255, 255], dtype="uint8")
    mask_raw = cv2.inRange(hsv, lower, upper)
    mask = mask_raw.copy()
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    # Keep the largest non-border component to suppress background artifacts.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels > 1:
        h, w = mask.shape
        best_label = None
        best_area = -1
        for label in range(1, num_labels):
            x = stats[label, cv2.CC_STAT_LEFT]
            y = stats[label, cv2.CC_STAT_TOP]
            bw = stats[label, cv2.CC_STAT_WIDTH]
            bh = stats[label, cv2.CC_STAT_HEIGHT]
            area = stats[label, cv2.CC_STAT_AREA]
            touches_border = x <= 1 or y <= 1 or (x + bw) >= (w - 2) or (y + bh) >= (h - 2)
            if touches_border:
                continue
            if area > best_area:
                best_area = area
                best_label = label
        if best_label is None:
            best_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = (labels == best_label).astype(np.uint8) * 255

    # If mask is too small, fall back to the raw mask (or no removal).
    area_ratio = float(np.count_nonzero(mask)) / float(mask.size)
    if area_ratio < 0.02:
        mask = mask_raw
        area_ratio = float(np.count_nonzero(mask)) / float(mask.size)
    if area_ratio < 0.02:
        # Still too small, keep original image to avoid all-white output.
        cv2.imwrite(path_to_clean_image, img)
        return

    img_out = img.copy()
    img_out[mask == 0] = 255
    cv2.imwrite(path_to_clean_image, img_out)

def resize(path_to_warped_image, path_to_warped_image_clean, path_to_warped_image_mini, path_to_warped_image_clean_mini, resize_value):
    pil_img = Image.open(path_to_warped_image)
    pil_img_clean = Image.open(path_to_warped_image_clean)
    pil_img.resize((resize_value, resize_value), resample=Image.NEAREST).save(path_to_warped_image_mini)
    pil_img_clean.resize((resize_value, resize_value), resample=Image.NEAREST).save(path_to_warped_image_clean_mini)

def filter_palm_lines(palmline_path, clean_mini_path):
    palm_lines = cv2.imread(palmline_path, cv2.IMREAD_GRAYSCALE)
    clean_img = cv2.imread(clean_mini_path)
    if palm_lines is None or clean_img is None:
        return

    hsv = cv2.cvtColor(clean_img, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 20, 80], dtype=np.uint8)
    upper = np.array([50, 255, 255], dtype=np.uint8)
    fg = cv2.inRange(hsv, lower, upper)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if num_labels > 1:
        h, w = fg.shape
        best_label = None
        best_area = -1
        for label in range(1, num_labels):
            x = stats[label, cv2.CC_STAT_LEFT]
            y = stats[label, cv2.CC_STAT_TOP]
            bw = stats[label, cv2.CC_STAT_WIDTH]
            bh = stats[label, cv2.CC_STAT_HEIGHT]
            area = stats[label, cv2.CC_STAT_AREA]
            touches_border = x <= 1 or y <= 1 or (x + bw) >= (w - 2) or (y + bh) >= (h - 2)
            if touches_border:
                continue
            if area > best_area:
                best_area = area
                best_label = label
        if best_label is None:
            best_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        palm_mask = (labels == best_label).astype(np.uint8) * 255
    else:
        palm_mask = fg

    # If mask is too small, fall back to using the full image to avoid wiping all lines.
    area_ratio = float(np.count_nonzero(palm_mask)) / float(palm_mask.size)
    if area_ratio < 0.02:
        palm_mask = np.ones_like(palm_mask, dtype=np.uint8) * 255
    else:
        # Erode slightly to avoid boundary leakage, but keep more edge pixels for longer lines.
        palm_mask = cv2.erode(palm_mask, np.ones((3, 3), np.uint8), iterations=1)

    # Debug: report raw palm-line pixels before filtering.
    raw_count = int(np.count_nonzero(palm_lines))

    # First, mask out anything outside palm region.
    masked = cv2.bitwise_and(palm_lines, palm_lines, mask=palm_mask)
    masked_count = int(np.count_nonzero(masked))

    # Then remove small/isolated components from masked lines.
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(masked, connectivity=8)
    if num_labels <= 1:
        cv2.imwrite(palmline_path, masked)
        print(
            "[palmtrace] filter_palm_lines pixels: raw=",
            raw_count,
            "masked=",
            masked_count,
            "cleaned=",
            int(np.count_nonzero(masked)),
        )
        return

    # Identify main component by area.
    main_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    main_cy, main_cx = centroids[main_label]

    cleaned = np.zeros_like(masked)
    min_area = max(10, int(stats[main_label, cv2.CC_STAT_AREA] * 0.01))
    max_dist = max(palm_mask.shape) * 0.5
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        cy, cx = centroids[label]
        dist = ((cy - main_cy) ** 2 + (cx - main_cx) ** 2) ** 0.5
        if dist > max_dist:
            continue
        cleaned[labels == label] = 255

    cv2.imwrite(palmline_path, cleaned)
    print(
        "[palmtrace] filter_palm_lines pixels: raw=",
        raw_count,
        "masked=",
        masked_count,
        "cleaned=",
        int(np.count_nonzero(cleaned)),
    )

def save_result(im, contents, resize_value, path_to_result):
    if im is None:
        print_error()
    else:
        plt.figure()
        heart_content_1, heart_content_2, head_content_1, head_content_2, life_content_1, life_content_2 = contents
        image_height, image_width = im.size
        fontsize = 12
        
        plt.tick_params(
            axis='both',
            which='both',
            bottom=False,
            left=False,
            labelbottom=False,
            labelleft=False
        )

        plt.imshow(im)
        plt.savefig(path_to_result, bbox_inches="tight", pad_inches=0)
        plt.close()

def print_error():
    print('Palm lines not properly detected! Please use another palm image.')
