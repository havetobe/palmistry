from PIL import Image, ImageDraw
import cv2
import mediapipe as mp
import numpy as np

def measure(path_to_warped_image_mini, lines):
    heart_thres_x = 0
    head_thres_x = 0
    life_thres_y = 0

    mp_hands = mp.solutions.hands
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
        image = cv2.flip(cv2.imread(path_to_warped_image_mini), 1)
        image_height, image_width, _ = image.shape

        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        hand_landmarks = results.multi_hand_landmarks[0]

        zero = hand_landmarks.landmark[mp_hands.HandLandmark(0).value].y
        one = hand_landmarks.landmark[mp_hands.HandLandmark(1).value].y
        five = hand_landmarks.landmark[mp_hands.HandLandmark(5).value].x
        nine = hand_landmarks.landmark[mp_hands.HandLandmark(9).value].x
        thirteen = hand_landmarks.landmark[mp_hands.HandLandmark(13).value].x

        heart_thres_x = image_width * (1 - (nine + (five - nine) * 2 / 5))
        head_thres_x = image_width * (1 - (thirteen + (nine - thirteen) / 3))
        life_thres_y = image_height * (one + (zero - one) / 3)

    im = Image.open(path_to_warped_image_mini)
    width = 3
    if (None in lines) or (len(lines) < 3):
        return None, None
    else:
        draw = ImageDraw.Draw(im)

        heart_line = lines[0]
        head_line = lines[1]
        life_line = lines[2]

        # Drop lines that fall mostly outside the palm region (based on skin mask).
        img_bgr = cv2.imread(path_to_warped_image_mini)
        if img_bgr is not None:
            hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
            lower = np.array([0, 20, 80], dtype=np.uint8)
            upper = np.array([50, 255, 255], dtype=np.uint8)
            palm_mask = cv2.inRange(hsv, lower, upper)
            palm_mask = cv2.morphologyEx(palm_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

            def line_inside_ratio(line):
                inside = 0
                for y, x, _, _ in line:
                    if 0 <= y < palm_mask.shape[0] and 0 <= x < palm_mask.shape[1]:
                        if palm_mask[y, x] > 0:
                            inside += 1
                return inside / max(1, len(line))

            ratios = [line_inside_ratio(heart_line), line_inside_ratio(head_line), line_inside_ratio(life_line)]
            for idx, ratio in enumerate(ratios):
                if ratio < 0.7:
                    if idx == 0:
                        heart_line = []
                    elif idx == 1:
                        head_line = []
                    else:
                        life_line = []

        heart_line_points = [tuple(reversed(l[:2])) for l in heart_line]
        heart_line_tip = heart_line_points[0]
        heart_content_1 = 'Love line governs all matters of the heart, including romance, friendship, and commitment.'
        if heart_line_tip[0] < heart_thres_x:
            heart_content_2 = 'Your Heart line is long, which means you will have long partnership with whom you love or care.'
        else:
            heart_content_2 = 'Your Heart line is short, which means you will meet various people and have a broad range of relationships throughout your life.'
        draw.line(heart_line_points, fill="red", width=width)

        head_line_points = [tuple(reversed(l[:2])) for l in head_line]
        head_line_tip = head_line_points[-1]
        head_content_1 = 'Head line tells us about our intellectual curiosities and pursuits.'
        if head_line_tip[0] > head_thres_x:
            head_content_2 = 'Your Head line is long, which means you will explore a broad range of topics throughout your life.'
        else:
            head_content_2 = 'Your Head line is short, which means you will be fascinated by one topic and dig deep into it.'
        draw.line(head_line_points, fill="green", width=width)

        life_line_points = [tuple(reversed(l[:2])) for l in life_line]
        life_line_tip = life_line_points[-1]
        life_content_1 = 'Life line reveals your experiences, vitality, and zest. Be careful, it has nothing to do with how long you will live!'
        if life_line_tip[1] > life_thres_y:
            life_content_2 = 'Your Life line is long, which means you tend to solve problems with other people rather than by yourself.'
        else:
            life_content_2 = 'Your Life line is short, which means you are independent and autonomous.'
        draw.line(life_line_points, fill="blue", width=width)

        # draw.line([(heart_thres_x, 0), (heart_thres_x, image_height)], fill="red")
        # draw.line([(head_thres_x, 0), (head_thres_x, image_height)], fill="green")
        # draw.line([(0, life_thres_y), (image_width, life_thres_y)], fill="blue")

        contents = [heart_content_1, heart_content_2, head_content_1, head_content_2, life_content_1, life_content_2]
        return im, contents
