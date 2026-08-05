import numpy as np
from sklearn.metrics import f1_score, accuracy_score


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    n_classes = y_prob.shape[1]
    ece_per_class = []

    for c in range(n_classes):
        y_binary = (y_true == c).astype(int)
        prob_c = y_prob[:, c]

        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece_c = 0.0

        for i in range(n_bins):
            mask = (prob_c >= bin_edges[i]) & (prob_c < bin_edges[i + 1])
            if mask.sum() == 0:
                continue
            bin_acc = y_binary[mask].mean()
            bin_conf = prob_c[mask].mean()
            ece_c += mask.sum() * abs(bin_acc - bin_conf)

        ece_per_class.append(ece_c / len(y_true))

    return np.mean(ece_per_class)
