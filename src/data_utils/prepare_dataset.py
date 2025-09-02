import os
import glob
import shutil
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import argparse

PROCESSED_DATA_DIR = 'data/processed'
TEST_SPLIT_SIZE = 0.15
VAL_SPLIT_SIZE = 0.15
RANDOM_SEED = 42

def split_dataset(synthetic_data_dir):
    if not os.path.isdir(synthetic_data_dir):
        print(f"Error: Directory '{synthetic_data_dir}' does not exist.")
        return

    if os.path.exists(PROCESSED_DATA_DIR):
        shutil.rmtree(PROCESSED_DATA_DIR)
        
    all_paths = glob.glob(os.path.join(synthetic_data_dir, '**', '*.png'), recursive=True)
    labels = [os.path.basename(os.path.dirname(p)) for p in all_paths]

    train_val_paths, test_paths, train_val_labels, _ = train_test_split(
        all_paths, labels, test_size=TEST_SPLIT_SIZE, stratify=labels, random_state=RANDOM_SEED
    )
    
    relative_val_size = VAL_SPLIT_SIZE / (1 - TEST_SPLIT_SIZE)
    train_paths, val_paths, _, _ = train_test_split(
        train_val_paths, train_val_labels, test_size=relative_val_size, stratify=train_val_labels, random_state=RANDOM_SEED
    )
    
    copy_files('train', train_paths)
    copy_files('val', val_paths)
    copy_files('test', test_paths)

def copy_files(split_name, paths):
    for path in tqdm(paths, desc=f'Copying {split_name} files'):
        class_name = os.path.basename(os.path.dirname(path))
        dest_dir = os.path.join(PROCESSED_DATA_DIR, split_name, class_name)
        os.makedirs(dest_dir, exist_ok=True)
        shutil.copy(path, dest_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Split balanced dataset.")
    parser.add_argument('--synthetic_data_dir', type=str, default='data/synthetic')
    args = parser.parse_args()
    split_dataset(args.synthetic_data_dir)