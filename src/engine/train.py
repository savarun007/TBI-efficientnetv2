import os
import glob
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm
from timm.data.mixup import Mixup

from src.data_utils.dataset import HemorrhageDataset, get_transforms
from src.models.get_model import get_model

def run_one_epoch(model, loader, criterion, optimizer, scaler, device, mixup_fn, is_training, epoch_desc):
    model.train() if is_training else model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    
    progress_bar = tqdm(loader, desc=epoch_desc)
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        
        if is_training:
            if mixup_fn:
                images, labels = mixup_fn(images, labels)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        else: # Validation
            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss = nn.CrossEntropyLoss()(outputs, labels)
                total_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
    avg_loss = total_loss / len(loader)
    val_f1 = f1_score(all_labels, all_preds, average='macro') if not is_training else 0.0
    return avg_loss, val_f1

def get_data_loaders(data_dir, image_size, batch_size, class_to_idx):
    train_paths = glob.glob(os.path.join(data_dir, 'train', '**', '*.png'), recursive=True)
    val_paths = glob.glob(os.path.join(data_dir, 'val', '**', '*.png'), recursive=True)
    train_labels = [os.path.basename(os.path.dirname(p)) for p in train_paths]
    val_labels = [os.path.basename(os.path.dirname(p)) for p in val_paths]
    train_dataset = HemorrhageDataset(train_paths, train_labels, class_to_idx, get_transforms('train', image_size))
    val_dataset = HemorrhageDataset(val_paths, val_labels, class_to_idx, get_transforms('val', image_size))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    class_names = sorted(os.listdir(os.path.join(args.data_dir, 'train')))
    num_classes = len(class_names)
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    model = get_model(args.model_name, num_classes=num_classes).to(device)
    scaler = torch.amp.GradScaler('cuda')
    best_f1, patience_counter = 0.0, 0

    if not args.progressive_resizing:
        print(f"--- Running Single-Stage Training for {args.epochs} epochs ---")
        train_loader, val_loader = get_data_loaders(args.data_dir, args.image_size, args.batch_size, class_to_idx)
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        mixup_fn = Mixup(mixup_alpha=args.mixup_alpha, cutmix_alpha=args.cutmix_alpha, label_smoothing=args.label_smoothing, num_classes=num_classes) if args.mixup_alpha > 0 or args.cutmix_alpha > 0 else None

        for epoch in range(args.epochs):
            train_loss, _ = run_one_epoch(model, train_loader, criterion, optimizer, scaler, device, mixup_fn, True, f"Epoch {epoch+1}/{args.epochs} [Train]")
            val_loss, val_f1 = run_one_epoch(model, val_loader, criterion, None, None, device, None, False, f"Epoch {epoch+1}/{args.epochs} [Val]")
            scheduler.step()
            print(f"Epoch {epoch+1}/{args.epochs} -> Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Macro F1: {val_f1:.4f}")
            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save(model.state_dict(), os.path.join(args.output_dir, f"{args.model_name}_best.pth"))
    else:
        print(f"\n--- STAGE 1: Training at {args.image_size_s1}x{args.image_size_s1} for {args.stage1_epochs} epochs ---")
        train_loader_s1, val_loader_s1 = get_data_loaders(args.data_dir, args.image_size_s1, args.batch_size, class_to_idx)
        optimizer = optim.AdamW(model.parameters(), lr=args.stage1_lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.stage1_epochs)
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        mixup_fn = Mixup(mixup_alpha=0.2, cutmix_alpha=1.0, label_smoothing=args.label_smoothing, num_classes=num_classes)

        for epoch in range(args.stage1_epochs):
            if epoch < args.freeze_epochs:
                for name, param in model.named_parameters():
                    if 'head' not in name and 'classifier' not in name: param.requires_grad = False
                if epoch == 0: print(f"Backbone frozen for {args.freeze_epochs} epochs.")
            elif epoch == args.freeze_epochs:
                for param in model.parameters(): param.requires_grad = True
                print(f"Backbone unfrozen.")
            train_loss, _ = run_one_epoch(model, train_loader_s1, criterion, optimizer, scaler, device, mixup_fn, True, f"S1 Epoch {epoch+1}/{args.stage1_epochs} [Train]")
            val_loss, val_f1 = run_one_epoch(model, val_loader_s1, criterion, None, None, device, None, False, f"S1 Epoch {epoch+1}/{args.stage1_epochs} [Val]")
            scheduler.step()
            print(f"S1 Epoch {epoch+1}/{args.stage1_epochs} -> Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Macro F1: {val_f1:.4f}")

            if val_f1 > best_f1:
                best_f1 = val_f1; torch.save(model.state_dict(), os.path.join(args.output_dir, f"{args.model_name}_best.pth")); patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= args.patience: break
        
        if patience_counter < args.patience:
            print(f"\n--- STAGE 2: Fine-tuning at {args.image_size_s2}x{args.image_size_s2} for {args.stage2_epochs} epochs ---")
            batch_size_s2 = max(1, args.batch_size // 2)
            train_loader_s2, val_loader_s2 = get_data_loaders(args.data_dir, args.image_size_s2, batch_size_s2, class_to_idx)
            optimizer = optim.AdamW(model.parameters(), lr=args.stage2_lr, weight_decay=args.weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.stage2_epochs)

            for epoch in range(args.stage2_epochs):
                train_loss, _ = run_one_epoch(model, train_loader_s2, criterion, optimizer, scaler, device, mixup_fn, True, f"S2 Epoch {epoch+1}/{args.stage2_epochs} [Train]")
                val_loss, val_f1 = run_one_epoch(model, val_loader_s2, criterion, None, None, device, None, False, f"S2 Epoch {epoch+1}/{args.stage2_epochs} [Val]")
                scheduler.step()
                print(f"S2 Epoch {epoch+1}/{args.stage2_epochs} -> Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Macro F1: {val_f1:.4f}")
                if val_f1 > best_f1:
                    best_f1 = val_f1; torch.save(model.state_dict(), os.path.join(args.output_dir, f"{args.model_name}_best.pth")); patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= args.patience: break
    print(f"\n--- Training complete. Best validation F1-score: {best_f1:.4f} ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Advanced training for TBI classification.")
    parser.add_argument('--model_name', type=str, required=True, choices=['efficientnetv2_s', 'swin_tiny', 'convnext_tiny'])
    parser.add_argument('--data_dir', type=str, default='data/processed')
    parser.add_argument('--output_dir', type=str, default='outputs/checkpoints')
    parser.add_argument('--weight_decay', type=float, default=0.05); parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--mixup_alpha', type=float, default=0.0); parser.add_argument('--cutmix_alpha', type=float, default=0.0)
    parser.add_argument('--patience', type=int, default=7)
    
    single_stage_group = parser.add_argument_group('Single-Stage Training'); single_stage_group.add_argument('--epochs', type=int, default=20)
    single_stage_group.add_argument('--image_size', type=int, default=224); single_stage_group.add_argument('--learning_rate', type=float, default=1e-4)
    single_stage_group.add_argument('--batch_size', type=int, default=32)

    prog_group = parser.add_argument_group('Progressive Resizing'); prog_group.add_argument('--progressive_resizing', action='store_true')
    prog_group.add_argument('--stage1_epochs', type=int, default=20); prog_group.add_argument('--image_size_s1', type=int, default=224)
    prog_group.add_argument('--stage1_lr', type=float, default=1e-3); prog_group.add_argument('--freeze_epochs', type=int, default=5)
    prog_group.add_argument('--stage2_epochs', type=int, default=15); prog_group.add_argument('--image_size_s2', type=int, default=260)
    prog_group.add_argument('--stage2_lr', type=float, default=5e-5)
    
    args = parser.parse_args()
    if not args.progressive_resizing: args.batch_size_s1 = args.batch_size
    train(args)