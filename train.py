"""
EEG Seizure Classification using University of Bonn Dataset
Binary classifier to distinguish seizure vs non-seizure EEG segments
"""

import os
import numpy as np
import scipy.signal
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import glob

# Create output directories
os.makedirs('images/roc', exist_ok=True)
os.makedirs('images/confusion_matrices', exist_ok=True)
os.makedirs('images/analysis', exist_ok=True)

# Constants
FS = 173.61  # Sampling rate in Hz
EPOCH_LENGTH = 1.0  # 1 second epochs
EPOCH_SAMPLES = int(EPOCH_LENGTH * FS)  # 173 samples
EPOCHS_PER_FILE = 20
RANDOM_SEED = 42

# Frequency bands (Hz)
FREQ_BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 70)
}

# Feature names for interpretability
FEATURE_NAMES = [
    'mean', 'std', 'rms', 'max', 'min', 'line_length', 'zero_crossing_rate',
    'delta_power', 'theta_power', 'alpha_power', 'beta_power', 'gamma_power'
]


def load_eeg_file(filepath):
    """Load EEG data from a text file."""
    data = np.loadtxt(filepath)
    return data


def extract_random_epochs(data, num_epochs=EPOCHS_PER_FILE, epoch_length=EPOCH_SAMPLES):
    """
    Extract random non-overlapping epochs from EEG data.
    
    Args:
        data: 1D array of EEG samples
        num_epochs: Number of epochs to extract
        epoch_length: Length of each epoch in samples
    
    Returns:
        Array of shape (num_epochs, epoch_length)
    """
    data_length = len(data)
    max_start = data_length - epoch_length
    
    if max_start < num_epochs:
        num_epochs = max(1, max_start // epoch_length)
    
    valid_starts = []
    attempts = 0
    max_attempts = max_start * 10
    
    while len(valid_starts) < num_epochs and attempts < max_attempts:
        start = np.random.randint(0, max_start + 1)
        if not any(abs(start - s) < epoch_length for s in valid_starts):
            valid_starts.append(start)
        attempts += 1
    
    if len(valid_starts) < num_epochs:
        remaining = num_epochs - len(valid_starts)
        step = max_start // (remaining + 1)
        for i in range(1, remaining + 1):
            new_start = i * step
            if not any(abs(new_start - s) < epoch_length for s in valid_starts):
                valid_starts.append(new_start)
    
    valid_starts = sorted(valid_starts)[:num_epochs]
    epochs = np.array([data[start:start+epoch_length] for start in valid_starts])
    
    return epochs


def extract_time_features(epoch):
    """Extract time-domain features from a single epoch."""
    mean = np.mean(epoch)
    std = np.std(epoch)
    rms = np.sqrt(np.mean(epoch ** 2))
    max_val = np.max(epoch)
    min_val = np.min(epoch)
    line_length = np.sum(np.abs(np.diff(epoch)))
    zero_crossings = np.sum(np.diff(np.sign(epoch)) != 0)
    zero_crossing_rate = zero_crossings / len(epoch)
    
    return [mean, std, rms, max_val, min_val, line_length, zero_crossing_rate]


def extract_frequency_features(epoch, fs=FS):
    """Extract frequency-domain features from a single epoch."""
    freqs, psd = scipy.signal.welch(epoch, fs=fs, nperseg=min(len(epoch), 128))
    
    band_powers = []
    for band_name, (low_freq, high_freq) in FREQ_BANDS.items():
        band_mask = (freqs >= low_freq) & (freqs <= high_freq)
        band_power = np.sum(psd[band_mask])
        band_powers.append(band_power)
    
    return band_powers


def extract_features(epoch):
    """Extract all features (time + frequency) from a single epoch."""
    time_feat = extract_time_features(epoch)
    freq_feat = extract_frequency_features(epoch)
    return np.array(time_feat + freq_feat)


def main():
    """Main function to train the EEG seizure classifier."""
    np.random.seed(RANDOM_SEED)
    
    print("Loading EEG data...")
    
    seizure_files = sorted(glob.glob('S*.txt'))
    non_seizure_files = sorted(glob.glob('Z*.txt'))
    
    print(f"Found {len(seizure_files)} seizure files and {len(non_seizure_files)} non-seizure files")
    
    all_features = []
    all_labels = []
    
    print("Processing seizure files...")
    for filepath in seizure_files:
        data = load_eeg_file(filepath)
        epochs = extract_random_epochs(data, num_epochs=EPOCHS_PER_FILE, epoch_length=EPOCH_SAMPLES)
        for epoch in epochs:
            features = extract_features(epoch)
            all_features.append(features)
            all_labels.append(1)
    
    print("Processing non-seizure files...")
    for filepath in non_seizure_files:
        data = load_eeg_file(filepath)
        epochs = extract_random_epochs(data, num_epochs=EPOCHS_PER_FILE, epoch_length=EPOCH_SAMPLES)
        for epoch in epochs:
            features = extract_features(epoch)
            all_features.append(features)
            all_labels.append(0)
    
    X = np.array(all_features)
    y = np.array(all_labels)
    
    print(f"Total samples: {len(X)}")
    print(f"Feature shape: {X.shape}")
    print(f"Class distribution: {np.bincount(y)}")
    
    print("\nSplitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )
    
    print(f"Train set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    print("\nTraining logistic regression model...")
    model = LogisticRegression(solver='liblinear', random_state=RANDOM_SEED, max_iter=1000)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    print(f"\nAUC: {roc_auc:.4f}")
    
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    
    best_threshold = thresholds[best_idx]
    best_tpr = tpr[best_idx]
    best_fpr = fpr[best_idx]
    
    print("\nOptimal threshold (Youden's J):", best_threshold)
    print("TPR at optimal threshold:", best_tpr)
    print("FPR at optimal threshold:", best_fpr)
    
    y_pred_opt = (y_prob >= best_threshold).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_opt).ravel()
    
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    
    print("\nPerformance at optimal threshold:")
    print("Sensitivity (TPR):", sensitivity)
    print("Specificity (TNR):", specificity)
    
    print("\nConfusion Matrix (at Youden-J optimal threshold):")
    print("=" * 50)
    print(f"True Normal (TN):     {tn}")
    print(f"False Positive (FP):  {fp}")
    print(f"False Negative (FN):  {fn}")
    print(f"True Positive (TP):   {tp}")
    print("=" * 50)
    print(f"\nMatrix format:")
    print(f"[[{tn:4d}  {fp:4d}]")
    print(f" [{fn:4d}  {tp:4d}]]")
    
    # Plot confusion matrix heatmap
    cm = np.array([[tn, fp], [fn, tp]])
    
    plt.figure(figsize=(8, 7))
    im = plt.imshow(cm, interpolation='nearest', cmap='Blues', aspect='auto')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    
    thresh = cm.max() / 2.
    for i in range(2):
        for j in range(2):
            plt.text(j, i, f'{cm[i, j]}',
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=16, fontweight='bold')
    
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.xticks([0, 1], ['Normal', 'Seizure'], fontsize=11)
    plt.yticks([0, 1], ['Normal', 'Seizure'], fontsize=11)
    plt.title(f'Confusion Matrix at Optimal Youden-J Threshold (τ = {best_threshold:.3f})',
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('images/confusion_matrices/confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("\nConfusion matrix saved to confusion_matrix.png")
    
    # Plot ROC curve
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
    plt.scatter(best_fpr, best_tpr, color='red', s=100, zorder=5,
                label=f'Optimal threshold = {best_threshold:.3f}')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve - EEG Seizure Classification', fontsize=14)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('images/roc/roc.png', dpi=300, bbox_inches='tight')
    print("ROC curve saved to roc.png")
    
    # Feature importance analysis
    print("\n" + "=" * 50)
    print("Feature Importance (Logistic Regression Coefficients)")
    print("=" * 50)
    
    coef = model.coef_[0]
    feature_importance = list(zip(FEATURE_NAMES, coef))
    feature_importance_sorted = sorted(feature_importance, key=lambda x: abs(x[1]), reverse=True)
    
    for name, weight in feature_importance_sorted:
        print(f"{name:22s}: {weight:+.4f}")
    
    # Plot feature importance bar chart
    plt.figure(figsize=(10, 6))
    
    sorted_idx = np.argsort(coef)
    sorted_names = [FEATURE_NAMES[i] for i in sorted_idx]
    sorted_coef = coef[sorted_idx]
    
    colors = ['#d73027' if c < 0 else '#1a9850' for c in sorted_coef]
    plt.barh(range(len(sorted_names)), sorted_coef, color=colors)
    
    plt.yticks(range(len(sorted_names)), sorted_names, fontsize=11)
    plt.xlabel('Coefficient Value (Seizure Prediction Weight)', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title('Feature Importance for Seizure Classification\n(Positive = Seizure Indicator, Negative = Non-Seizure Indicator)', 
              fontsize=13, fontweight='bold')
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('images/analysis/feature_importance.png', dpi=300, bbox_inches='tight')
    print("\nFeature importance plot saved to feature_importance.png")
    
    print(f"\nAUC: {roc_auc:.2f}")


if __name__ == '__main__':
    main()
