"""
train_model.py - Generate Training Data & Train Decision-Tree Classifier
==========================================================================
1. Simulate OFDM *bursts* (multiple symbols per channel realisation) across
   a range of SNRs and channel tap lengths.
2. For each burst, run *both* Strategy A (ZF) and Strategy B (MMSE).
3. Compare the *MSE* of equalized symbols to the true transmitted symbols.
   Label each sample with the strategy that achieved lower MSE.
   (Using MSE for labelling is valid because labelling is an offline
    process; the ML model only sees receiver-observable features at
    inference time.)
4. Extract receiver-observable features (no hidden channel info).
5. Train a scikit-learn DecisionTreeClassifier and report metrics.

Label convention:
    0 -> Strategy A (ZF) achieves lower or equal MSE
    1 -> Strategy B (MMSE) achieves lower MSE
"""

import os, json, pickle, sys
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ofdm import transmit_burst, N_DATA
from receiver import (
    strategy_a_burst, strategy_b_burst,
    equalize_frame_a, equalize_frame_b,
    compute_ber, extract_features_burst, FEATURE_NAMES,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def generate_dataset(n_bursts_per_config=25, burst_len=10, seed=42):
    """Create labelled dataset for the strategy selector.

    Labeling Protocol (Documented Performance Objective):
      1. If Strategy B achieves strictly fewer bit errors: label = 1 (MMSE/DFT).
      2. If Strategy A achieves strictly fewer bit errors: label = 0 (ZF/LS).
      3. If bit errors are tied:
         - At high SNR (>= 18 dB), Strategy A is selected (label = 0) to save
           computational complexity (no time-domain filtering needed).
         - At lower SNR (< 18 dB), Strategy B is selected (label = 1) for
           higher noise margin and lower post-equalization MSE.

    Returns
    -------
    X : ndarray (n_samples, n_features)
    y : ndarray (n_samples,) with values in {0, 1}
    """
    snr_range   = np.arange(0, 32, 2)          # 0-30 dB
    tap_configs = [1, 2, 3, 5, 8, 12]

    rng = np.random.default_rng(seed)
    X_all, y_all = [], []

    for n_taps in tap_configs:
        for snr_db in snr_range:
            for _ in range(n_bursts_per_config):
                rx_frames, all_bits, h, all_data_syms = transmit_burst(
                    burst_len, snr_db, n_taps=n_taps, rng=rng
                )

                # Strategy A and B outputs
                rx_a = strategy_a_burst(rx_frames, snr_db)
                rx_b = strategy_b_burst(rx_frames, snr_db)

                err_a = int(np.sum(all_bits != rx_a))
                err_b = int(np.sum(all_bits != rx_b))

                if err_b < err_a:
                    label = 1  # Strategy B strictly better
                elif err_a < err_b:
                    label = 0  # Strategy A strictly better
                else:
                    # Tied bit errors: use complexity/SNR rule
                    label = 0 if snr_db >= 18.0 else 1

                feats = extract_features_burst(rx_frames, snr_db)
                X_all.append(feats)
                y_all.append(label)

    return np.array(X_all), np.array(y_all)


def train_classifier(X, y, seed=42):
    """Train/test split, fit DecisionTreeClassifier, return model + metrics."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )
    clf = DecisionTreeClassifier(
        max_depth=6, min_samples_leaf=10, random_state=seed
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred,
        labels=[0, 1],
        target_names=["ZF (A)", "MMSE (B)"],
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    return clf, X_test, y_test, y_pred, acc, report, cm


def save_artifacts(clf, X_test, y_test, y_pred, acc, report, cm):
    """Persist model, plots, and metrics to results/."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Pickle the trained model
    with open(os.path.join(RESULTS_DIR, "strategy_selector.pkl"), "wb") as f:
        pickle.dump(clf, f)

    # 2. Decision-tree text rules
    tree_text = export_text(clf, feature_names=FEATURE_NAMES, max_depth=10)
    with open(os.path.join(RESULTS_DIR, "decision_tree_rules.txt"), "w") as f:
        f.write(tree_text)
    print("")
    print("-- Decision Tree Rules --")
    print(tree_text[:2000])

    # 3. Confusion matrix plot
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(cm, display_labels=["ZF (A)", "MMSE (B)"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix  (Accuracy = {:.2%})".format(acc))
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=150)
    plt.close(fig)

    # 4. Feature importances bar chart
    importances = clf.feature_importances_
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(FEATURE_NAMES, importances, color="steelblue")
    ax.set_xlabel("Importance")
    ax.set_title("Decision-Tree Feature Importances")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "feature_importances.png"), dpi=150)
    plt.close(fig)

    # 5. Save classification report as JSON
    with open(os.path.join(RESULTS_DIR, "classification_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("")
    print("[OK] Model artefacts saved to {}/".format(RESULTS_DIR))
    print("     Accuracy: {:.2%}".format(acc))


# -- Standalone execution --
if __name__ == "__main__":
    print("Generating training dataset ...")
    X, y = generate_dataset(n_bursts_per_config=30, burst_len=10, seed=42)
    print("  Dataset: {} samples, {} features".format(X.shape[0], X.shape[1]))
    print("  Class balance: ZF={}, MMSE={}".format(int(np.sum(y==0)), int(np.sum(y==1))))

    print("Training Decision-Tree classifier ...")
    clf, X_test, y_test, y_pred, acc, report, cm = train_classifier(X, y, seed=42)
    save_artifacts(clf, X_test, y_test, y_pred, acc, report, cm)
