import os
import glob
import argparse
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
from tqdm import tqdm

from src.data_utils.dataset import HemorrhageDataset, get_transforms
from src.models.get_model import get_model

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = os.path.join(args.checkpoint_dir, f"{args.model_name}_best.pth")
    if not os.path.exists(checkpoint_path): return

    test_paths = glob.glob(os.path.join(args.data_dir, 'test', '**', '*.png'), recursive=True)
    class_names = sorted(os.listdir(os.path.join(args.data_dir, 'test')))
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    test_labels = [os.path.basename(os.path.dirname(p)) for p in test_paths]

    test_transforms = get_transforms('val', args.image_size)
    test_dataset = HemorrhageDataset(test_paths, test_labels, class_to_idx, test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = get_model(args.model_name, num_classes=len(class_names), pretrained=False).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc=f"Evaluating {args.model_name}"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(f"\n--- Evaluation Report for {args.model_name} ---")
    print(f"Overall Test Accuracy: {accuracy_score(all_labels, all_preds) * 100:.2f}%")
    print("\nClassification Report:"); print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))
    
    results_df = pd.DataFrame({'true_label': [class_names[i] for i in all_labels], 'predicted_label': [class_names[i] for i in all_preds]})
    results_path = os.path.join(args.output_dir, f"{args.model_name}_predictions.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Predictions saved to: {results_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a model.")
    parser.add_argument('--model_name', type=str, required=True, choices=['efficientnetv2_s', 'swin_tiny', 'convnext_tiny'])
    parser.add_argument('--data_dir', type=str, default='data/processed')
    parser.add_argument('--checkpoint_dir', type=str, default='outputs/checkpoints')
    parser.add_argument('--output_dir', type=str, default='outputs/plots')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()
    evaluate(args)