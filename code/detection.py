import numpy as np
from PIL import Image
import torch
def detect(net, jpeg_dir, output_dir, resize_value, device=torch.device('cpu')):
    pil_img = Image.open(jpeg_dir)
    img = np.asarray(pil_img.resize((resize_value, resize_value), resample=Image.NEAREST)) / 255
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).permute(0,3,1,2).to(device)
    pred = net(img).squeeze(0)
    pred = torch.sigmoid(pred)
    prob = pred.detach().cpu().numpy()[0]
    mask = (prob > 0.04).astype(np.uint8)
    if mask.sum() < 30:
        # If thresholding wipes everything, fall back to a top-percentile mask.
        thresh = float(np.percentile(prob, 92))
        mask = (prob >= thresh).astype(np.uint8)
    mask = (mask * 255).astype(np.uint8)
    Image.fromarray(mask).convert("RGB").save(output_dir)
