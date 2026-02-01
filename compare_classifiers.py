"""
Multi-Classifier Comparison for EEG Seizure Classification
Compares Logistic Regression, Random Forest, SVM, and Gradient Boosting
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_curve, auc, confusion_matrix, accuracy_score, 
                            f1_score, brier_score_loss)
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve

# Import functions from train.py
from train import (
    load_eeg_file, extract_random_epochs, extract_features,
    RANDOM_SEED, EPOCHS_PER_FILE, EPOCH_SAMPLES
)
import glob

# Create output directories
os.makedirs('images/roc', exist_ok=True)
os.makedirs('images/confusion_matrices', exist_ok=True)
os.makedirs('images/analysis', exist_ok=True)


def setup_classifiers():
    """Initialize classifier instances."""
    classifiers = {
        'Logistic Regression': LogisticRegression(
            solver='liblinear', 
            random_state=RANDOM_SEED, 
            max_iter=1000
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=RANDOM_SEED,
            n_jobs=-1
        ),
        'SVM': SVC(
            kernel='rbf',
            probability=True,
            random_state=RANDOM_SEED,
            C=1.0,
            gamma='scale'
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=RANDOM_SEED
        )
    }
    return classifiers


def compute_certainty_metrics(y_test, y_prob, y_pred):
    """Compute certainty-related metrics."""
    # Average confidence: mean of maximum probability for each prediction
    max_probs = np.maximum(y_prob, 1 - y_prob)  # Confidence in predicted class
    avg_confidence = np.mean(max_probs)
    
    # High-confidence predictions (confidence > 0.8)
    high_conf_mask = max_probs > 0.8
    high_conf_count = np.sum(high_conf_mask)
    high_conf_pct = (high_conf_count / len(y_test)) * 100
    
    if high_conf_count > 0:
        high_conf_accuracy = accuracy_score(y_test[high_conf_mask], y_pred[high_conf_mask])
    else:
        high_conf_accuracy = 0.0
    
    # Brier score (calibration error) - lower is better
    brier_score = brier_score_loss(y_test, y_prob)
    
    # Confidence distribution statistics
    conf_std = np.std(max_probs)
    conf_min = np.min(max_probs)
    conf_max = np.max(max_probs)
    conf_median = np.median(max_probs)
    
    return {
        'avg_confidence': avg_confidence,
        'high_conf_accuracy': high_conf_accuracy,
        'high_conf_pct': high_conf_pct,
        'brier_score': brier_score,
        'conf_std': conf_std,
        'conf_min': conf_min,
        'conf_max': conf_max,
        'conf_median': conf_median,
        'max_probs': max_probs
    }


def train_classifier(name, model, X_train, y_train, X_test, y_test):
    """Train classifier and return performance metrics."""
    print(f"  Training {name}...")
    
    try:
        # Train model
        model.fit(X_train, y_train)
        
        # Get predictions and probabilities
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]  # Probability of class 1 (seizure)
        
        # Compute ROC curve and AUC
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        # Compute Youden's J optimal threshold
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        best_threshold = thresholds[best_idx]
        
        # Predictions at optimal threshold
        y_pred_opt = (y_prob >= best_threshold).astype(int)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_opt).ravel()
        
        # Performance metrics
        accuracy = accuracy_score(y_test, y_pred_opt)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = f1_score(y_test, y_pred_opt)
        
        # Certainty metrics
        certainty = compute_certainty_metrics(y_test, y_prob, y_pred_opt)
        
        print(f"    {name} completed - AUC: {roc_auc:.4f}, Avg Confidence: {certainty['avg_confidence']:.4f}")
        
        return {
            'name': name,
            'model': model,
            'roc_auc': roc_auc,
            'accuracy': accuracy,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'f1_score': f1,
            'best_threshold': best_threshold,
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'y_test': y_test,
            'y_prob': y_prob,
            'y_pred': y_pred_opt,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'tp': tp,
            **certainty
        }
    
    except Exception as e:
        print(f"    ERROR training {name}: {str(e)}")
        return None


def load_data():
    """load and prepare EEG data."""
    np.random.seed(RANDOM_SEED)
    
    print("loading EEG data...")
    
    seizure_files = sorted(glob.glob('S*.txt'))
    non_seizure_files = sorted(glob.glob('Z*.txt'))
    
    print(f"Found {len(seizure_files)} seizure files and {len(non_seizure_files)} non-seizure files")
    
    all_features = []
    all_labels = []
    
    print("processing seizure files...")
    for filepath in seizure_files:
        data = load_eeg_file(filepath)
        epochs = extract_random_epochs(data, num_epochs=EPOCHS_PER_FILE, epoch_length=EPOCH_SAMPLES)
        for epoch in epochs:
            features = extract_features(epoch)
            all_features.append(features)
            all_labels.append(1)
    
    print("processing non-seizure files...")
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
    
    # Train/test split
    print("\nsplitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )
    
    print(f"Train set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    return X_train, X_test, y_train, y_test


def generate_comparison_matrix(results):
    """Generate comparison matrix."""
    data = []
    for r in results:
        if r is not None:
            data.append({
                'Classifier': r['name'],
                'AUC': r['roc_auc'],
                'Accuracy': r['accuracy'],
                'Sensitivity': r['sensitivity'],
                'Specificity': r['specificity'],
                'F1-Score': r['f1_score'],
                'Avg Confidence': r['avg_confidence'],
                'High Conf Acc': r['high_conf_accuracy'],
                'High Conf %': r['high_conf_pct'],
                'Brier Score': r['brier_score']
            })
    
    return data


def plot_comparison_heatmap(data):
    """Generate comparison heatmap."""
    # Prepare data for heatmap
    metrics = ['AUC', 'Accuracy', 'Sensitivity', 'Specificity', 'F1-Score', 
               'Avg Confidence', 'High Conf Acc', 'High Conf %']
    
    n_classifiers = len(data)
    n_metrics = len(metrics) + 1  # +1 for Brier Score
    
    # Extract metric values
    heatmap_data = np.zeros((n_classifiers, n_metrics))
    classifier_names = []
    
    for i, row in enumerate(data):
        classifier_names.append(row['Classifier'])
        for j, metric in enumerate(metrics):
            heatmap_data[i, j] = row[metric]
        # Brier Score (inverted so higher is better)
        heatmap_data[i, -1] = 1 - row['Brier Score']
    
    # Normalize each column to [0, 1]
    normalized_data = heatmap_data.copy()
    for j in range(n_metrics):
        col_min = normalized_data[:, j].min()
        col_max = normalized_data[:, j].max()
        if col_max > col_min:
            normalized_data[:, j] = (normalized_data[:, j] - col_min) / (col_max - col_min)
    
    metric_labels = metrics + ['Brier Score (inverted)']
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    im = ax.imshow(normalized_data.T, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
    
    ax.set_xticks(range(n_classifiers))
    ax.set_xticklabels(classifier_names, rotation=45, ha='right')
    ax.set_yticks(range(n_metrics))
    ax.set_yticklabels(metric_labels)
    
    for i in range(n_classifiers):
        for j in range(n_metrics):
            if j < len(metrics):
                orig_value = heatmap_data[i, j]
                text = f'{orig_value:.4f}'
            else:
                # Brier Score (show original, not inverted)
                text = f'{data[i]["Brier Score"]:.4f}'
            ax.text(i, j, text, ha='center', va='center', 
                   color='white' if normalized_data[i, j] < 0.5 else 'black',
                   fontsize=9, fontweight='bold')
    
    plt.colorbar(im, ax=ax, label='Normalized Score (Green=Best, Red=Worst)')
    plt.title('Classifier Comparison Matrix\n(Green=Best Performance, Red=Worst Performance)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('images/analysis/classifier_comparison.png', dpi=300, bbox_inches='tight')
    print("Comparison heatmap saved to classifier_comparison.png")


def plot_roc_comparison(results):
    """Plot overlaid ROC curves."""
    plt.figure(figsize=(10, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, r in enumerate(results):
        if r is not None:
            plt.plot(r['fpr'], r['tpr'], 
                    label=f"{r['name']} (AUC = {r['roc_auc']:.4f})",
                    linewidth=2, color=colors[i % len(colors)])
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve Comparison - All Classifiers', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('images/roc/roc_comparison.png', dpi=300, bbox_inches='tight')
    print("ROC comparison plot saved to roc_comparison.png")


def plot_individual_roc_curves(results):
   
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, r in enumerate(results):
        if r is not None:
            plt.figure(figsize=(8, 8))
            
            # Plot ROC curve
            plt.plot(r['fpr'], r['tpr'], 
                    linewidth=3, color=colors[i % len(colors)],
                    label=f'{r["name"]} (AUC = {r["roc_auc"]:.4f})')
            
            plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
            
           
            if 'thresholds' not in r:
                from sklearn.metrics import roc_curve
                _, _, thresholds = roc_curve(r['y_test'], r['y_prob'])
                r['thresholds'] = thresholds
            
            j_scores = r['tpr'] - r['fpr']
            best_idx = np.argmax(j_scores)
            best_fpr = r['fpr'][best_idx]
            best_tpr = r['tpr'][best_idx]
            best_threshold = r.get('best_threshold', r['thresholds'][best_idx] if 'thresholds' in r else 0.5)
            
            plt.scatter(best_fpr, best_tpr, color='red', s=150, zorder=5,
                       marker='*', label=f'Optimal Threshold = {best_threshold:.3f}')
            
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
            plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
            plt.title(f'ROC Curve - {r["name"]}\nAUC = {r["roc_auc"]:.4f}', 
                     fontsize=14, fontweight='bold')
            plt.legend(loc="lower right", fontsize=11)
            plt.grid(alpha=0.3)
            plt.tight_layout()
            
            filename = 'images/roc/' + r['name'].lower().replace(' ', '_') + '_roc.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Individual ROC curve saved to {filename}")
            plt.close()


def plot_individual_confusion_matrices(results):
    """Plot confusion matrices for each classifier."""
    for r in results:
        if r is not None:
            cm = np.array([[r['tn'], r['fp']], [r['fn'], r['tp']]])
            
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
            
            title = (f'Confusion Matrix - {r["name"]}\n'
                    f'Optimal Threshold (Youden-J) = {r["best_threshold"]:.3f}\n'
                    f'Sensitivity: {r["sensitivity"]:.4f} | Specificity: {r["specificity"]:.4f} | Accuracy: {r["accuracy"]:.4f}')
            plt.title(title, fontsize=13, fontweight='bold', pad=20)
            
            plt.tight_layout()
            
            filename = 'images/confusion_matrices/' + r['name'].lower().replace(' ', '_') + '_confusion_matrix.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Individual confusion matrix saved to {filename}")
            plt.close()


def plot_certainty_distributions(results):
    """Plot confidence distribution histograms."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, r in enumerate(results):
        if r is not None:
            ax = axes[i]
            max_probs = r['max_probs']
            
            ax.hist(max_probs, bins=30, alpha=0.7, color=f'C{i}', edgecolor='black')
            ax.axvline(r['avg_confidence'], color='red', linestyle='--', linewidth=2,
                      label=f'Mean: {r["avg_confidence"]:.3f}')
            ax.axvline(r['conf_median'], color='green', linestyle='--', linewidth=2,
                      label=f'Median: {r["conf_median"]:.3f}')
            
            ax.set_xlabel('Prediction Confidence', fontsize=11)
            ax.set_ylabel('Frequency', fontsize=11)
            ax.set_title(f'{r["name"]}\nAvg Conf: {r["avg_confidence"]:.4f}, '
                        f'High Conf Acc: {r["high_conf_accuracy"]:.4f}',
                        fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('images/analysis/certainty_analysis.png', dpi=300, bbox_inches='tight')
    print("Certainty analysis plot saved to certainty_analysis.png")


def plot_calibration_curves(results):
    """Plot probability calibration curves."""
    plt.figure(figsize=(10, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, r in enumerate(results):
        if r is not None:
            try:
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    r['y_test'], r['y_prob'], n_bins=10
                )
                plt.plot(mean_predicted_value, fraction_of_positives, 
                        marker='o', linewidth=2, label=f"{r['name']} (Brier={r['brier_score']:.4f})",
                        color=colors[i % len(colors)])
            except Exception as e:
                print(f"  Warning: Could not plot calibration for {r['name']}: {e}")
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect Calibration')
    plt.xlabel('Mean Predicted Probability', fontsize=12)
    plt.ylabel('Fraction of Positives', fontsize=12)
    plt.title('Probability Calibration Curves', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('images/analysis/calibration_curves.png', dpi=300, bbox_inches='tight')
    print("Calibration curves plot saved to calibration_curves.png")


def identify_best_classifier(data):
    """Identify best classifier using composite score."""
    best_score = -1
    best_classifier = None
    best_metrics = None
    
    for row in data:
        # Weighted score: AUC (40%), Accuracy (20%), Avg Confidence (20%), F1 (10%), Brier (10%)
        composite_score = (
            0.4 * row['AUC'] +
            0.2 * row['Accuracy'] +
            0.2 * row['Avg Confidence'] +
            0.1 * row['F1-Score'] +
            0.1 * (1 - row['Brier Score'])  # lower brier is better
        )
        row['Composite Score'] = composite_score
        
        if composite_score > best_score:
            best_score = composite_score
            best_classifier = row['Classifier']
            best_metrics = row.copy()
    
    return best_classifier, best_metrics


def main():
    """Compare all classifiers."""
    print("Multi-Classifier Comparison for EEG Seizure Classification")
    print("=" * 70)
    
    # load data
    X_train, X_test, y_train, y_test = load_data()
    
    # set up classifiers
    print("\n" + "=" * 70)
    print("Setting up classifiers...")
    classifiers = setup_classifiers()
    print(f"  Configured {len(classifiers)} classifiers")
    
    # train all classifiers
    print("\n" + "=" * 70)
    print("Training all classifiers...")
    results = []
    for name, model in classifiers.items():
        result = train_classifier(name, model, X_train, y_train, X_test, y_test)
        results.append(result)
    
    # fillter out None results
    results = [r for r in results if r is not None]
    
    if len(results) == 0:
        print("ERROR: No classifiers trained successfully")
        return
    
    # make comparison matrix
    print("\n" + "=" * 70)
    print("Making comparison matrix...")
    comparison_data = generate_comparison_matrix(results)
    
    # print comparison table
    print("\n" + "=" * 70)
    print("\nClassifier Comparison Matrix")
    print("-" * 70)
    
    headers = ['Classifier', 'AUC', 'Accuracy', 'Sensitivity', 'Specificity', 
               'F1-Score', 'Avg Conf', 'High Conf Acc', 'High Conf %', 'Brier Score']
    col_widths = [20, 8, 10, 12, 12, 10, 10, 14, 12, 12]
    
    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    print(header_line)
    print("-" * len(header_line))
    
    for row in comparison_data:
        values = [
            row['Classifier'],
            f"{row['AUC']:.4f}",
            f"{row['Accuracy']:.4f}",
            f"{row['Sensitivity']:.4f}",
            f"{row['Specificity']:.4f}",
            f"{row['F1-Score']:.4f}",
            f"{row['Avg Confidence']:.4f}",
            f"{row['High Conf Acc']:.4f}",
            f"{row['High Conf %']:.1f}%",
            f"{row['Brier Score']:.4f}"
        ]
        data_line = " | ".join(v.ljust(w) for v, w in zip(values, col_widths))
        print(data_line)
    
    print("=" * 70)
    
    # Identify best classifier
    best_name, best_metrics = identify_best_classifier(comparison_data)
    print(f"\nBest Classifier: {best_name}")
    print(f"   AUC: {best_metrics['AUC']:.4f}")
    print(f"   Accuracy: {best_metrics['Accuracy']:.4f}")
    print(f"   Average Confidence: {best_metrics['Avg Confidence']:.4f}")
    print(f"   Composite Score: {best_metrics['Composite Score']:.4f}")
    
    # Generate visualizations
    print("\n" + "=" * 70)
    plot_comparison_heatmap(comparison_data)
    plot_roc_comparison(results)
    plot_individual_roc_curves(results)
    plot_individual_confusion_matrices(results)
    plot_certainty_distributions(results)
    plot_calibration_curves(results)
    
    print("\n" + "=" * 70)
    print("All outputs saved.")
    print("=" * 70)


if __name__ == '__main__':
    main()
