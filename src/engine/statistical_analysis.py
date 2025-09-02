import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
import argparse
from scipy.stats import norm

def analyze_and_graph(args):
    results_path = os.path.join(args.results_dir, f"{args.model_name}_predictions.csv")
    if not os.path.exists(results_path): return
        
    df = pd.read_csv(results_path)
    true_labels = df['true_label']
    pred_labels = df['predicted_label']
    class_names = sorted(df['true_label'].unique())
    num_classes = len(class_names)
    accuracy = accuracy_score(true_labels, pred_labels)
    kappa = cohen_kappa_score(true_labels, pred_labels)
    n = len(true_labels)
    
    print(f"\n--- Statistical Analysis Report for {args.model_name} ---")
    z_ci = 1.96
    margin_of_error = z_ci * np.sqrt((accuracy * (1 - accuracy)) / n)
    print(f"\n[95% Confidence Interval for Accuracy]"); print(f"Observed Accuracy (p̂): {accuracy:.4f}"); print(f"95% CI: [{accuracy - margin_of_error:.4f}, {accuracy + margin_of_error:.4f}]")
    print(f"\n[Cohen's Kappa Coefficient (κ)]"); print(f"Cohen's Kappa: {kappa:.4f}")
    
    p0 = 1 / num_classes
    z_statistic = (accuracy - p0) / np.sqrt((p0 * (1 - p0)) / n)
    alpha = 0.05
    critical_value = norm.ppf(1 - alpha)
    print(f"\n[One-Proportion Z-Test (vs. Random Chance)]"); print(f"Z-statistic: {z_statistic:.2f}"); print(f"Critical Value at α=0.05: {critical_value:.2f}")
    if z_statistic > critical_value: print("Conclusion: REJECT the null hypothesis (model is significant).")
    else: print("Conclusion: FAIL to reject the null hypothesis.")
    
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(true_labels, pred_labels, labels=class_names)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label'); plt.ylabel('True Label'); plt.title(f'Confusion Matrix for {args.model_name}')
    plt.savefig(os.path.join(args.results_dir, f"{args.model_name}_confusion_matrix.png"))
    print("\nGraphs saved.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Perform statistical analysis.")
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--results_dir', type=str, default='outputs/plots')
    args = parser.parse_args()
    analyze_and_graph(args)