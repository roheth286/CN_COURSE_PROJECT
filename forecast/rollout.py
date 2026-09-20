"""
NetForecaster - Multi-Step Recursive Forecasting Engine (Stage 6)

This module implements:
1. RolloutResult dataclass storing autoregressive forecast trajectories.
2. Autoregressive Rollout: Recursive multi-step prediction over K horizons (k=1..K).
   In each step k:
     a. The World Model predicts next state S_hat_{t+k}, threat risk P(Attack_{t+k}),
        and attack category C_hat_{t+k}.
     b. The historical context window rolls forward: oldest state is dropped,
        and newly predicted S_hat_{t+k} is appended.
     c. The process repeats recursively for horizons k=1..K.
3. Pure NumPy metric evaluation functions (AUC, F1, Precision, Recall, Accuracy, MSE).
4. Full horizon evaluation suite measuring forecasting accuracy across horizons k=1..K.
5. Inverse state unscaling to recover physical network metrics from normalized vectors.
"""

import os
import sys
import json
import dataclasses
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
import torch
import torch.nn as nn

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.network_world_model import NetworkWorldModel
from features.windowing import STATE_FEATURE_NAMES, STATE_VECTOR_DIM
from features.sequence_dataset import NetworkStateScaler
from data.preprocess import INDEX_TO_CLASS, CLASS_TO_INDEX


# ==============================================================================
# 1. Rollout Result Container
# ==============================================================================

@dataclass
class RolloutResult:
    """
    Structured container for multi-step autoregressive rollout predictions.

    Attributes:
    -----------
    predicted_states : np.ndarray
        Array of shape [batch_size, horizon, state_dim] with normalized state forecasts.
    risk_probabilities : np.ndarray
        Array of shape [batch_size, horizon] with attack risk probabilities in [0, 1].
    category_probabilities : np.ndarray
        Array of shape [batch_size, horizon, num_classes] with softmax class probabilities.
    category_predictions : np.ndarray
        Array of shape [batch_size, horizon] with predicted class integer indices.
    horizon : int
        Number of recursive forecast steps (K).
    state_feature_names : List[str]
        Ordered names of the 36 network state features.
    """
    predicted_states: np.ndarray
    risk_probabilities: np.ndarray
    category_probabilities: np.ndarray
    category_predictions: np.ndarray
    horizon: int
    state_feature_names: List[str] = dataclasses.field(
        default_factory=lambda: list(STATE_FEATURE_NAMES)
    )

    def unscale_states(self, scaler: NetworkStateScaler) -> np.ndarray:
        """
        Transform predicted normalized states back to physical network units
        (e.g., packets/sec, bytes/sec, port entropies, TCP flag rates).

        Parameters:
        -----------
        scaler : NetworkStateScaler
            Fitted scaler from Stage 4.

        Returns:
        --------
        np.ndarray of shape [batch_size, horizon, state_dim] in original physical units.
        """
        batch_size, horizon, state_dim = self.predicted_states.shape
        # Flatten batch and horizon to 2D for inverse transform
        reshaped = self.predicted_states.reshape(-1, state_dim)
        unscaled_flat = scaler.inverse_transform(reshaped)
        # Reshape back to [batch_size, horizon, state_dim]
        return unscaled_flat.reshape(batch_size, horizon, state_dim)

    def to_dict(self) -> Dict[str, Any]:
        """Convert container to dictionary with serializable lists."""
        return {
            "predicted_states": self.predicted_states.tolist(),
            "risk_probabilities": self.risk_probabilities.tolist(),
            "category_probabilities": self.category_probabilities.tolist(),
            "category_predictions": self.category_predictions.tolist(),
            "horizon": self.horizon,
            "state_feature_names": self.state_feature_names,
        }

    def save_npz(self, filepath: str) -> None:
        """Save rollout arrays to a compressed .npz archive."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        np.savez_compressed(
            filepath,
            predicted_states=self.predicted_states,
            risk_probabilities=self.risk_probabilities,
            category_probabilities=self.category_probabilities,
            category_predictions=self.category_predictions,
            horizon=np.array([self.horizon]),
        )

    @classmethod
    def load_npz(cls, filepath: str) -> "RolloutResult":
        """Load rollout arrays from a compressed .npz archive."""
        data = np.load(filepath)
        horizon = int(data["horizon"][0])
        return cls(
            predicted_states=data["predicted_states"],
            risk_probabilities=data["risk_probabilities"],
            category_probabilities=data["category_probabilities"],
            category_predictions=data["category_predictions"],
            horizon=horizon,
        )


# ==============================================================================
# 2. Autoregressive Rollout Engine
# ==============================================================================

def autoregressive_rollout(
    model: NetworkWorldModel,
    initial_history: Union[torch.Tensor, np.ndarray],
    horizon: int = 5,
    device: Union[str, torch.device] = "cpu"
) -> RolloutResult:
    """
    Perform multi-step recursive forecasting using the trained Network World Model.

    Algorithm:
    ----------
    For step k = 1 to horizon:
      1. Forward pass on current sliding context:
         [pred_state, risk_logits, cat_logits] = model(current_history)
      2. Calibrate risk probability via Sigmoid and category probabilities via Softmax.
      3. Append predicted state to the history and drop the oldest state:
         current_history = [current_history[:, 1:, :], pred_state.unsqueeze(1)]
      4. Repeat recursively for step k+1.

    Parameters:
    -----------
    model : NetworkWorldModel
        Trained PyTorch World Model.
    initial_history : torch.Tensor or np.ndarray
        Historical sequence of m state vectors.
        Shape: [batch_size, seq_len_m, state_dim] or [seq_len_m, state_dim].
    horizon : int, default=5
        Number of steps to forecast into the future (K).
        With 30-second windows, K=5 corresponds to 2.5 minutes lookahead.
    device : str or torch.device, default='cpu'
        Computation device.

    Returns:
    --------
    RolloutResult:
        Container holding predicted states, risk probabilities, and category predictions.
    """
    model.to(device)
    model.eval()

    # Convert input to PyTorch tensor if needed
    if isinstance(initial_history, np.ndarray):
        history_tensor = torch.tensor(initial_history, dtype=torch.float32)
    else:
        history_tensor = initial_history.clone().float()

    # Handle unbatched single sequence [seq_len, state_dim] -> [1, seq_len, state_dim]
    is_unbatched = (history_tensor.ndim == 2)
    if is_unbatched:
        history_tensor = history_tensor.unsqueeze(0)

    history_tensor = history_tensor.to(device)
    batch_size, seq_len, state_dim = history_tensor.shape

    # Storage for predictions across all forecast horizons
    all_pred_states: List[torch.Tensor] = []
    all_risk_probs: List[torch.Tensor] = []
    all_cat_probs: List[torch.Tensor] = []
    all_cat_preds: List[torch.Tensor] = []

    current_window = history_tensor.clone()

    with torch.no_grad():
        for step in range(horizon):
            # 1. Forward pass on the current sliding window
            pred_state, risk_logits, cat_logits = model(current_window)

            # 2. Calibrate risk probability via Sigmoid
            risk_prob = torch.sigmoid(risk_logits).squeeze(-1)  # [batch_size]

            # 3. Compute category probabilities and greedy class prediction
            cat_probs = torch.softmax(cat_logits, dim=-1)        # [batch_size, num_classes]
            cat_pred = torch.argmax(cat_probs, dim=-1)           # [batch_size]

            # 4. Save step predictions
            all_pred_states.append(pred_state)
            all_risk_probs.append(risk_prob)
            all_cat_probs.append(cat_probs)
            all_cat_preds.append(cat_pred)

            # 5. Recursive state shift: drop oldest state, append newly predicted state
            # current_window[:, 1:, :] has shape [batch_size, seq_len - 1, state_dim]
            # pred_state.unsqueeze(1) has shape [batch_size, 1, state_dim]
            next_state_slice = pred_state.unsqueeze(1)
            current_window = torch.cat([current_window[:, 1:, :], next_state_slice], dim=1)

    # Stack results along the horizon dimension [batch_size, horizon, ...]
    pred_states_arr = torch.stack(all_pred_states, dim=1).cpu().numpy()
    risk_probs_arr = torch.stack(all_risk_probs, dim=1).cpu().numpy()
    cat_probs_arr = torch.stack(all_cat_probs, dim=1).cpu().numpy()
    cat_preds_arr = torch.stack(all_cat_preds, dim=1).cpu().numpy()

    return RolloutResult(
        predicted_states=pred_states_arr,
        risk_probabilities=risk_probs_arr,
        category_probabilities=cat_probs_arr,
        category_predictions=cat_preds_arr,
        horizon=horizon,
    )


# ==============================================================================
# 3. Pure NumPy Evaluation Metrics (Zero External DLL Dependencies)
# ==============================================================================

def calculate_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute binary classification metrics without external library dependencies:
    Accuracy, Precision, Recall, F1-Score, False Positive Rate (FPR), and ROC-AUC.
    """
    y_true = np.asarray(y_true, dtype=np.int32).flatten()
    y_prob = np.asarray(y_prob, dtype=np.float32).flatten()
    y_pred = (y_prob >= threshold).astype(np.int32)

    total_samples = len(y_true)
    if total_samples == 0:
        return {
            "accuracy": 0.0, "precision": 0.0, "recall": 0.0,
            "f1": 0.0, "fpr": 0.0, "roc_auc": 0.5
        }

    accuracy = float(np.mean(y_true == y_pred))

    tp = float(np.sum((y_true == 1) & (y_pred == 1)))
    fp = float(np.sum((y_true == 0) & (y_pred == 1)))
    fn = float(np.sum((y_true == 1) & (y_pred == 0)))
    tn = float(np.sum((y_true == 0) & (y_pred == 0)))

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    # ROC-AUC via rank-sum (Mann-Whitney U statistic)
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        roc_auc = 0.5
    else:
        # Sort samples by predicted probability
        order = np.argsort(y_prob)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, total_samples + 1, dtype=np.float64)

        # Handle ties in probabilities with average rank
        unique_probs, inverse_indices, counts = np.unique(
            y_prob, return_inverse=True, return_counts=True
        )
        for i, count in enumerate(counts):
            if count > 1:
                tied_mask = (inverse_indices == i)
                ranks[tied_mask] = np.mean(ranks[tied_mask])

        pos_rank_sum = np.sum(ranks[y_true == 1])
        u_stat = pos_rank_sum - (n_pos * (n_pos + 1)) / 2.0
        roc_auc = float(u_stat / (n_pos * n_neg))
        roc_auc = max(0.0, min(1.0, roc_auc))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "roc_auc": roc_auc,
    }


def calculate_multiclass_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 8
) -> Dict[str, Any]:
    """
    Compute multi-class accuracy and macro-averaged F1 score.
    """
    y_true = np.asarray(y_true, dtype=np.int32).flatten()
    y_pred = np.asarray(y_pred, dtype=np.int32).flatten()

    total_samples = len(y_true)
    if total_samples == 0:
        return {"accuracy": 0.0, "macro_f1": 0.0, "per_class_f1": {}}

    accuracy = float(np.mean(y_true == y_pred))

    per_class_f1: Dict[str, float] = {}
    f1_list: List[float] = []

    for c in range(num_classes):
        class_name = INDEX_TO_CLASS.get(c, f"Class_{c}")
        tp = float(np.sum((y_true == c) & (y_pred == c)))
        fp = float(np.sum((y_true != c) & (y_pred == c)))
        fn = float(np.sum((y_true == c) & (y_pred != c)))

        # Precision and Recall
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        # Only include in macro-average if the class appears in ground truth
        if np.sum(y_true == c) > 0:
            f1_list.append(f1)

        per_class_f1[class_name] = float(f1)

    macro_f1 = float(np.mean(f1_list)) if f1_list else 0.0

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1,
    }


# ==============================================================================
# 4. Multi-Horizon Evaluation Suite
# ==============================================================================

def evaluate_rollout_horizons(
    model: NetworkWorldModel,
    test_sequences_path: str,
    horizon: int = 5,
    device: Union[str, torch.device] = "cpu",
    batch_size: int = 64
) -> Dict[str, Any]:
    """
    Evaluate the autoregressive rollout on test sequences across all horizons k=1..K.

    Calculates:
    - Threat Risk Detection Performance (Acc, Precision, Recall, F1, FPR, ROC-AUC) per horizon.
    - Attack Category Attribution Performance (Accuracy, Macro-F1) per horizon.
    - Physical State Forecast Error (MSE, MAE) at horizon k=1.

    Parameters:
    -----------
    model : NetworkWorldModel
        Trained model instance.
    test_sequences_path : str
        Path to test_sequences.npz.
    horizon : int, default=5
        Forecast horizon (K).
    device : str or torch.device, default='cpu'
        Computation device.
    batch_size : int, default=64
        Batch size for inference.

    Returns:
    --------
    Dict[str, Any] detailing performance metrics across horizons k=1..K.
    """
    if not os.path.exists(test_sequences_path):
        raise FileNotFoundError(f"Test sequences file not found at: {test_sequences_path}")

    # Load test archive
    data = np.load(test_sequences_path)
    x_test = data["x"]                  # [N, 10, 36]
    y_state_test = data["y_state"]      # [N, 36]
    y_binary_test = data["y_binary"]    # [N, 5]
    y_class_test = data["y_class"]      # [N, 5]

    num_samples = len(x_test)
    num_batches = int(np.ceil(num_samples / batch_size))

    all_pred_states: List[np.ndarray] = []
    all_risk_probs: List[np.ndarray] = []
    all_cat_probs: List[np.ndarray] = []
    all_cat_preds: List[np.ndarray] = []

    # Run rollout in mini-batches
    for b in range(num_batches):
        start_idx = b * batch_size
        end_idx = min(start_idx + batch_size, num_samples)
        batch_x = x_test[start_idx:end_idx]

        rollout_res = autoregressive_rollout(
            model=model,
            initial_history=batch_x,
            horizon=horizon,
            device=device,
        )

        all_pred_states.append(rollout_res.predicted_states)
        all_risk_probs.append(rollout_res.risk_probabilities)
        all_cat_probs.append(rollout_res.category_probabilities)
        all_cat_preds.append(rollout_res.category_predictions)

    pred_states = np.concatenate(all_pred_states, axis=0)       # [N, K, 36]
    risk_probs = np.concatenate(all_risk_probs, axis=0)         # [N, K]
    cat_probs = np.concatenate(all_cat_probs, axis=0)           # [N, K, 8]
    cat_preds = np.concatenate(all_cat_preds, axis=0)           # [N, K]

    # Evaluate per-horizon metrics
    horizon_results: Dict[str, Any] = {}
    actual_k = min(horizon, y_binary_test.shape[1])

    for k in range(1, actual_k + 1):
        step_idx = k - 1
        lookahead_seconds = k * 30

        # Ground truth targets for horizon step k
        gt_binary_k = y_binary_test[:, step_idx]
        gt_class_k = y_class_test[:, step_idx]

        # Forecasted outputs for horizon step k
        pred_risk_k = risk_probs[:, step_idx]
        pred_cat_k = cat_preds[:, step_idx]

        risk_metrics = calculate_binary_metrics(gt_binary_k, pred_risk_k)
        cat_metrics = calculate_multiclass_metrics(gt_class_k, pred_cat_k, num_classes=8)

        horizon_info: Dict[str, Any] = {
            "horizon_step": k,
            "lookahead_seconds": lookahead_seconds,
            "lookahead_minutes": round(lookahead_seconds / 60.0, 2),
            "risk_metrics": risk_metrics,
            "category_metrics": cat_metrics,
        }

        # For k=1, calculate state transition reconstruction error
        if k == 1:
            pred_state_k1 = pred_states[:, 0, :]
            mse = float(np.mean((pred_state_k1 - y_state_test) ** 2))
            mae = float(np.mean(np.abs(pred_state_k1 - y_state_test)))
            horizon_info["state_metrics"] = {
                "state_mse": mse,
                "state_mae": mae,
            }

        horizon_results[f"horizon_k{k}"] = horizon_info

    # Aggregate summary
    summary: Dict[str, Any] = {
        "num_test_samples": num_samples,
        "max_horizon": horizon,
        "window_duration_seconds": 30,
        "horizons": horizon_results,
    }

    return summary


def unscale_state_trajectories(
    predicted_states: np.ndarray,
    scaler: NetworkStateScaler
) -> np.ndarray:
    """
    Helper function to unscale multi-step normalized state trajectories.
    """
    batch_size, horizon, state_dim = predicted_states.shape
    reshaped = predicted_states.reshape(-1, state_dim)
    unscaled_flat = scaler.inverse_transform(reshaped)
    return unscaled_flat.reshape(batch_size, horizon, state_dim)
