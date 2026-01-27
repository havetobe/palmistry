import os
import argparse
from tools import *
from model import *
from rectification import *
from detection import *
from classification import *
from measurement import *

def resolve_input_path(input_arg, base_dir):
    if os.path.isabs(input_arg) and os.path.exists(input_arg):
        return input_arg
    if os.path.exists(input_arg):
        return os.path.abspath(input_arg)
    for folder in ("input", "inputs"):
        candidate = os.path.join(base_dir, folder, input_arg)
        if os.path.exists(candidate):
            return candidate
    for folder in ("input", "inputs"):
        candidate = os.path.join(os.getcwd(), folder, input_arg)
        if os.path.exists(candidate):
            return candidate
    return os.path.join(base_dir, "input", input_arg)

def main(input):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path_to_input_image = resolve_input_path(input, base_dir)

    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    resize_value = 256
    path_to_clean_image = os.path.join(results_dir, 'palm_without_background.jpg')
    path_to_warped_image = os.path.join(results_dir, 'warped_palm.jpg')
    path_to_warped_image_clean = os.path.join(results_dir, 'warped_palm_clean.jpg')
    path_to_warped_image_mini = os.path.join(results_dir, 'warped_palm_mini.jpg')
    path_to_warped_image_clean_mini = os.path.join(results_dir, 'warped_palm_clean_mini.jpg')
    path_to_palmline_image = os.path.join(results_dir, 'palm_lines.png')
    path_to_model = os.path.join(base_dir, 'checkpoint', 'checkpoint_aug_epoch70.pth')
    path_to_result = os.path.join(results_dir, 'result.jpg')

    # 0. Preprocess image
    remove_background(path_to_input_image, path_to_clean_image)

    # 1. Palm image rectification
    warp_result = warp(path_to_input_image, path_to_warped_image)
    if warp_result is None:
        print_error()
    else:
        remove_background(path_to_warped_image, path_to_warped_image_clean)
        resize(path_to_warped_image, path_to_warped_image_clean, path_to_warped_image_mini, path_to_warped_image_clean_mini, resize_value)

        # 2. Principal line detection
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'[INFO] Torch device: {device}')
        net = UNet(n_channels=3, n_classes=1).to(device)
        net.load_state_dict(torch.load(path_to_model, map_location=device, weights_only=True))
        detect(net, path_to_warped_image_clean, path_to_palmline_image, resize_value, device=device)
        filter_palm_lines(path_to_palmline_image, path_to_warped_image_clean_mini)

        # 3. Line classification
        lines = classify(path_to_palmline_image, path_to_warped_image_clean_mini)

        # 4. Length measurement
        im, contents = measure(path_to_warped_image_mini, lines)

        # 5. Save result
        save_result(im, contents, resize_value, path_to_result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='the path to the input')
    args = parser.parse_args()
    main(args.input)
