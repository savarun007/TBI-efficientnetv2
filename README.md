TBI Classification with EfficientNetV2-S
This project develops and evaluates a state-of-the-art deep learning model for the multi-class classification of Traumatic Brain Hemorrhages from head CT scan images. This guide details the complete pipeline for replicating the best-performing model, EfficientNetV2-S.

Features
Data Balancing: Automated generation of a large, perfectly balanced dataset from an imbalanced source using data augmentation.

Advanced Model Training: A script to train the EfficientNetV2-S model using a multi-stage strategy with progressive resizing.

Rigorous Evaluation: Automated generation of detailed classification reports, including per-class precision, recall, and F1-scores.

Statistical Validation: Scripts to perform statistical tests (95% Confidence Interval, Cohen's Kappa, Z-Test) to validate the significance of model performance.

Explainability (XAI): Generation of Grad-CAM++ visualizations to interpret model predictions.

Publication-Ready Figures: Standalone scripts to generate high-quality plots for research papers.

Project Structure
brain_hemorrhage_classification/
├── data/
│ ├── raw/ <-- Place your original 6 class folders here
│ ├── synthetic/ <-- Created by Step 2.1
│ └── processed/ <-- Created by Step 2.2
├── outputs/
│ ├── checkpoints/
│ ├── logs/
│ └── plots/
├── paper_figures/
└── src/
├── data_utils/
├── engine/
├── models/
└── xai/
└── requirements.txt

1. Setup and Installation
   Prerequisites
   Python 3.9+

An NVIDIA GPU with CUDA drivers installed

Step 1: Clone the Repository
git clone <your-repository-url>
cd brain_hemorrhage_classification

Step 2: Create and Activate a Virtual Environment

# Create the environment

python -m venv venv

# Activate it (on Windows)

.\venv\Scripts\activate

# Activate it (on MacOS/Linux)

source venv/bin/activate

Step 3: Install Dependencies
First, install the CPU-based libraries from the requirements.txt file.

pip install -r requirements.txt

Step 4: Install PyTorch with GPU Support (Critical)
The standard pip install often installs a CPU-only version. To use your GPU, you must install the correct version from the official PyTorch website.

Go to: https://pytorch.org/get-started/locally/

Select: Stable, your OS, Pip, Python, and the latest CUDA version.

Run the generated command. It will look similar to this:

pip3 install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)

Step 5: Verify GPU Setup
Run this command to confirm PyTorch can see your GPU.

python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}')"

You must see GPU Available: True. If not, revisit the previous step.

2. Data Preparation Workflow
   Step 2.1: Place Original Data
   Place your original, unbalanced dataset folders (e.g., epidural, ne, subdural) inside the data/raw/ directory.

Step 2.2: Generate the Balanced Synthetic Dataset
This script will read your raw data and create a new, perfectly balanced dataset in data/synthetic with 10,000 images per class. This will take a significant amount of time and disk space.

python -m src.data_utils.generate_synthetic_data --raw_data_dir data/raw

Step 2.3: Split the Dataset for Training
This script takes the balanced synthetic data and splits it into train, val, and test sets inside data/processed.

python -m src.data_utils.prepare_dataset --synthetic_data_dir data/synthetic

3. EfficientNetV2-S: Training and Evaluation
   This workflow details the end-to-end process for the EfficientNetV2-S model.

Step 3.1: Train the Model
This command runs the advanced multi-stage training with progressive resizing. It will train first at 224x224, then fine-tune at 260x260.

python -m src.engine.train --model_name efficientnetv2_s --progressive_resizing --stage1_epochs 20 --stage2_epochs 15 --freeze_epochs 5 --image_size_s1 224 --image_size_s2 260 --stage1_lr 1e-3 --stage2_lr 5e-5 --batch_size 48

Step 3.2: Evaluate the Model
After training is complete, run the evaluation script to get the final test accuracy and save the predictions. The --image_size must be 260 to match the final training stage.

python -m src.engine.evaluate --model_name efficientnetv2_s --image_size 260

Step 3.3: Run Statistical Analysis
This script performs the confidence interval and Kappa tests and generates the associated graphs for the efficientnetv2_s results.

python -m src.engine.statistical_analysis --model_name efficientnetv2_s

Step 3.4: Generate XAI Visualizations (Optional)
Create Grad-CAM++ heatmaps to see what the model is focusing on. Use the final image size of 260.

python -m src.xai.run_xai --model_name efficientnetv2_s --image_size 260
