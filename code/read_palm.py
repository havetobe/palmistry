import os
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from tools import *
from model import *
from rectification import *
from detection import *
from classification import *
from measurement import *


def map_line_to_original(line, homography, warped_size, orig_size, src_size):
    warped_w, warped_h = warped_size
    orig_w, orig_h = orig_size
    src_w, src_h = src_size
    inv = np.linalg.inv(homography)
    mapped = []
    for pt in line:
        y_src, x_src = pt[0], pt[1]
        xw = x_src * warped_w / max(1, src_w)
        yw = y_src * warped_h / max(1, src_h)
        vec = np.array([xw, yw, 1.0], dtype=np.float64)
        src = inv @ vec
        if src[2] == 0:
            continue
        xf = src[0] / src[2]
        yf = src[1] / src[2]
        x_unflip = warped_w - xf
        y_unflip = yf
        mapped.append((x_unflip, y_unflip))
    return mapped


def extend_polyline(points, ratio=0.2, min_px=12.0, max_px=60.0):
    if len(points) < 2:
        return points
    p0 = points[-2]
    p1 = points[-1]
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    seg_len = (dx * dx + dy * dy) ** 0.5
    if seg_len == 0:
        return points
    total_len = 0.0
    for i in range(1, len(points)):
        vx = points[i][0] - points[i - 1][0]
        vy = points[i][1] - points[i - 1][1]
        total_len += (vx * vx + vy * vy) ** 0.5
    extend_len = max(min_px, min(max_px, total_len * ratio))
    ux = dx / seg_len
    uy = dy / seg_len
    new_point = (p1[0] + extend_len * ux, p1[1] + extend_len * uy)
    return points + [new_point]


def draw_lines_on_original(path_to_input_image, lines, homography, warped_size, output_path, src_size):
    im = ImageOps.exif_transpose(Image.open(path_to_input_image)).convert("RGB")
    draw = ImageDraw.Draw(im)
    width = 6
    colors = [(255, 0, 0), (0, 128, 0), (0, 0, 255)]
    for idx, line in enumerate(lines[:3]):
        if line is None:
            continue
        points = map_line_to_original(line, homography, warped_size, im.size, src_size)
        if idx == 2:
            points = extend_polyline(points)
        if len(points) < 2:
            continue
        draw.line(points, fill=colors[idx], width=width)
    im.save(output_path)

def main(input):
    path_to_input_image = 'input/{}'.format(input)

    results_dir = './results'
    os.makedirs(results_dir, exist_ok=True)

    resize_value = 256
    path_to_clean_image = 'results/palm_without_background.jpg'
    path_to_warped_image = 'results/warped_palm.jpg'
    path_to_warped_image_clean = 'results/warped_palm_clean.jpg'
    path_to_warped_image_mini = 'results/warped_palm_mini.jpg'
    path_to_warped_image_clean_mini = 'results/warped_palm_clean_mini.jpg'
    path_to_palmline_image = 'results/palm_lines.png'
    path_to_model = 'checkpoint/checkpoint_aug_epoch70.pth'
    path_to_result = 'results/result.jpg'
    path_to_keypoints = 'results/keypoints.json'

    # 0. Preprocess image
    remove_background(path_to_input_image, path_to_clean_image)

    # 1. Palm image rectification
    warp_result, homography, warped_size = warp_with_matrix(
        path_to_input_image, path_to_warped_image
    )
    if warp_result is None:
        print_error()
    else:
        remove_background(path_to_warped_image, path_to_warped_image_clean)
        resize(path_to_warped_image, path_to_warped_image_clean, path_to_warped_image_mini, path_to_warped_image_clean_mini, resize_value)

        keypoints = extract_semantic_keypoints(path_to_input_image)
        save_keypoints(keypoints, path_to_keypoints)

        # 2. Principal line detection
        net = UNet(n_channels=3, n_classes=1)
        net.load_state_dict(torch.load(path_to_model, map_location=torch.device('cpu')))
        detect(net, path_to_warped_image_clean, path_to_palmline_image, resize_value)

        # 3. Line classification
        lines = classify(path_to_palmline_image)

        # 4. Save result on the original image
        src_size = (resize_value, resize_value)
        draw_lines_on_original(
            path_to_input_image, lines, homography, warped_size, path_to_result, src_size
        )
        annotate_keypoints_on_image(path_to_result, keypoints)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='the path to the input')
    args = parser.parse_args()
    main(args.input)
