import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import cv2
from pillow_heif import register_heif_opener
import json

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
    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(img, img, mask=mask)
    b, g, r = cv2.split(result)  
    filter = g.copy()
    ret, mask = cv2.threshold(filter, 10, 255, 1)
    img[mask == 255] = 255
    cv2.imwrite(path_to_clean_image, img)

def resize(path_to_warped_image, path_to_warped_image_clean, path_to_warped_image_mini, path_to_warped_image_clean_mini, resize_value):
    pil_img = Image.open(path_to_warped_image)
    pil_img_clean = Image.open(path_to_warped_image_clean)
    pil_img.resize((resize_value, resize_value), resample=Image.NEAREST).save(path_to_warped_image_mini)
    pil_img_clean.resize((resize_value, resize_value), resample=Image.NEAREST).save(path_to_warped_image_clean_mini)

def save_result(im, contents, resize_value, path_to_result):
    if im is None:
        print_error()
    else:
        heart_content_1, heart_content_2, head_content_1, head_content_2, life_content_1, life_content_2 = contents
        image_height, image_width = im.size
        fontsize = 12
        
        plt.tick_params(
            axis='both',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            left=False,         # ticks along the top edge are off
            labelbottom=False,
            labelleft=False
        )

        plt.imshow(im)
        plt.axis('off')
        plt.savefig(path_to_result, bbox_inches="tight", pad_inches=0)

def print_error():
    print('Palm lines not properly detected! Please use another palm image.')

def extract_semantic_keypoints(path_to_image):
    # Approximate semantic points from MediaPipe hand landmarks.
    import mediapipe as mp

    image = cv2.flip(cv2.imread(path_to_image), 1)
    if image is None:
        return None

    image_height, image_width, _ = image.shape
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks is None:
            return None

        hand_landmarks = results.multi_hand_landmarks[0]
        def to_xy(idx):
            lm = hand_landmarks.landmark[idx]
            return [lm.x * image_width, lm.y * image_height]

        wrist = to_xy(mp_hands.HandLandmark.WRIST.value)
        thumb_cmc = to_xy(mp_hands.HandLandmark.THUMB_CMC.value)
        thumb_mcp = to_xy(mp_hands.HandLandmark.THUMB_MCP.value)
        index_mcp = to_xy(mp_hands.HandLandmark.INDEX_FINGER_MCP.value)
        index_pip = to_xy(mp_hands.HandLandmark.INDEX_FINGER_PIP.value)
        middle_mcp = to_xy(mp_hands.HandLandmark.MIDDLE_FINGER_MCP.value)
        ring_mcp = to_xy(mp_hands.HandLandmark.RING_FINGER_MCP.value)
        pinky_mcp = to_xy(mp_hands.HandLandmark.PINKY_MCP.value)

        # Weighted center to push palm_center slightly upward from the wrist.
        palm_center = [
            (0.10 * wrist[0] + 0.225 * index_mcp[0] + 0.25 * middle_mcp[0] + 0.225 * ring_mcp[0] + 0.20 * pinky_mcp[0]),
            (0.10 * wrist[1] + 0.225 * index_mcp[1] + 0.25 * middle_mcp[1] + 0.225 * ring_mcp[1] + 0.20 * pinky_mcp[1]),
        ]
        # Tiger mouth near the gap between thumb base and index base.
        # Move the tiger_mouth toward thumb root by shifting from index_mcp toward thumb_cmc.
        # Tiger mouth: midpoint of thumb CMC and index MCP,
        # then a stronger shift toward thumb side and a smaller shift upward.
        def _norm(vx, vy):
            n = (vx * vx + vy * vy) ** 0.5
            if n == 0:
                return 0.0, 0.0
            return vx / n, vy / n

        # Shift closer to thumb IP (landmark 3) per feedback.
        thumb_ip = to_xy(mp_hands.HandLandmark.THUMB_IP.value)
        thumb_tip = to_xy(mp_hands.HandLandmark.THUMB_TIP.value)
        # Shift further toward thumb IP/tip (3/4).
        base_x = 0.50 * thumb_cmc[0] + 0.10 * index_mcp[0] + 0.25 * thumb_ip[0] + 0.15 * thumb_tip[0]
        base_y = 0.50 * thumb_cmc[1] + 0.10 * index_mcp[1] + 0.25 * thumb_ip[1] + 0.15 * thumb_tip[1]

        # Use handedness to stabilize the thumb direction.
        handedness_label = "Unknown"
        if results.multi_handedness:
            handedness_label = results.multi_handedness[0].classification[0].label

        v_thumb_x = thumb_cmc[0] - index_mcp[0]
        v_thumb_y = thumb_cmc[1] - index_mcp[1]
        v_up_x = index_pip[0] - index_mcp[0]
        v_up_y = index_pip[1] - index_mcp[1]

        utx, uty = _norm(v_thumb_x, v_thumb_y)
        uux, uuy = _norm(v_up_x, v_up_y)

        # Ensure thumb direction points left for Left hand and right for Right hand.
        if handedness_label == "Left" and utx > 0:
            utx, uty = -utx, -uty
        elif handedness_label == "Right" and utx < 0:
            utx, uty = -utx, -uty

        thumb_dist = (v_thumb_x * v_thumb_x + v_thumb_y * v_thumb_y) ** 0.5
        tiger_mouth = [
            base_x + 0.08 * thumb_dist * uux,
            base_y + 0.08 * thumb_dist * uuy,
        ]
        palm_root = wrist

        all_landmarks = [to_xy(i) for i in range(21)]

        return {
            "palm_root": palm_root,
            "tiger_mouth": tiger_mouth,
            "palm_center": palm_center,
            "image_size": [image_width, image_height],
            "flipped": True,
            "handedness": handedness_label,
            "all_landmarks": all_landmarks,
            "landmarks": {
                "wrist": wrist,
                "thumb_cmc": thumb_cmc,
                "thumb_mcp": thumb_mcp,
                "index_mcp": index_mcp,
                "index_pip": index_pip,
                "middle_mcp": middle_mcp,
                "ring_mcp": ring_mcp,
                "pinky_mcp": pinky_mcp,
            },
        }

def save_keypoints(keypoints, path_to_json):
    if keypoints is None:
        return
    with open(path_to_json, "w", encoding="utf-8") as f:
        json.dump(keypoints, f, ensure_ascii=False, indent=2)

def annotate_keypoints_on_image(path_to_image, keypoints):
    if keypoints is None:
        return
    im = Image.open(path_to_image)
    draw = ImageDraw.Draw(im)
    r = 16
    colors = {
        "掌根": (255, 69, 0),   # orange-red
        "虎口": (0, 191, 255),  # deep sky blue
        "掌心": (0, 200, 0),    # green
    }
    img_w, img_h = im.size
    src_w, src_h = keypoints.get("image_size", [img_w, img_h])
    scale_x = img_w / src_w if src_w else 1.0
    scale_y = img_h / src_h if src_h else 1.0
    flipped = keypoints.get("flipped", False)
    labels = [
        ("掌根", keypoints.get("palm_root")),
        ("虎口", keypoints.get("tiger_mouth")),
        ("掌心", keypoints.get("palm_center")),
    ]
    for label, pt in labels:
        if pt is None:
            continue
        color = colors.get(label, (255, 69, 0))
        x, y = pt[0] * scale_x, pt[1] * scale_y
        if flipped:
            x = img_w - x
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color, outline=color, width=3)
        draw.line((x - r, y, x + r, y), fill=color, width=3)
        draw.line((x, y - r, x, y + r), fill=color, width=3)
        draw.text((x + r + 4, y - r - 6), label, fill=color)

    # Overlay MediaPipe 21 landmark indices for precise tuning.
    lm_color = (0, 0, 0)
    lm_r = 3
    all_landmarks = keypoints.get("all_landmarks", [])
    for i, pt in enumerate(all_landmarks):
        if pt is None:
            continue
        x, y = pt[0] * scale_x, pt[1] * scale_y
        if flipped:
            x = img_w - x
        draw.ellipse((x - lm_r, y - lm_r, x + lm_r, y + lm_r), fill=lm_color, outline=lm_color)
        draw.text((x + 4, y - 6), str(i), fill=lm_color)
    im.save(path_to_image)
