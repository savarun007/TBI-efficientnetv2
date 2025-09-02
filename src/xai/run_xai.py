import os
import glob
import random
import cv2
import torch
import numpy as np
from tqdm import tqdm
import argparse

from src.models.get_model import get_model
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

def generate_xai(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = os.path.join(args.checkpoint_dir, f"{args.model_name}_best.pth")
    if not os.path.exists(checkpoint_path): return
    
    class_names = sorted(os.listdir(os.path.join(args.data_dir, 'test')))
    model = get_model(args.model_name, num_classes=len(class_names), pretrained=False).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()

    target_layers = [model.conv_head] if 'efficientnet' in args.model_name else [model.stages[-1].blocks[-1].norm] if 'convnext' in args.model_name else [model.layers[-1].blocks[-1].norm2]
    cam = GradCAMPlusPlus(model=model, target_layers=target_layers)
    
    output_dir = os.path.join(args.output_dir, f"{args.model_name}_gradcam++")
    os.makedirs(output_dir, exist_ok=True)

    for class_name in class_names:
        class_dir = os.path.join(args.data_dir, 'test', class_name)
        image_paths = glob.glob(os.path.join(class_dir, '*.png'))
        sample_paths = random.sample(image_paths, min(len(image_paths), args.num_samples))
        
        for i, img_path in enumerate(tqdm(sample_paths, desc=f"Generating for '{class_name}'")):
            rgb_img = cv2.imread(img_path, 1)[:, :, ::-1]
            rgb_img = cv2.resize(rgb_img, (args.image_size, args.image_size))
            input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).to(device)
            grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
            visualization = show_cam_on_image(rgb_img / 255.0, grayscale_cam, use_rgb=True)
            save_path = os.path.join(output_dir, f"{class_name}_sample_{i+1}.png")
            cv2.imwrite(save_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate Grad-CAM++ visualizations.")
    parser.add_argument('--model_name', type=str, required=True, choices=['efficientnetv2_s', 'swin_tiny', 'convnext_tiny'])
    parser.add_argument('--image_size', type=int, required=True)
    parser.add_argument('--data_dir', type=str, default='data/processed')
    parser.add_argument('--checkpoint_dir', type=str, default='outputs/checkpoints')
    parser.add_argument('--output_dir', type=str, default='outputs/xai_results')
    parser.add_argument('--num_samples', type=int, default=4)
    args = parser.parse_args()
    generate_xai(args)