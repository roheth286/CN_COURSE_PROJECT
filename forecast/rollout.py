import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Any

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
import torch.nn as nn

from model.network_world_model import NetworkWorldModel
from features.sequence_dataset import NetworkStateScaler, STATE_FEATURE_NAMES, STATE_VECTOR_DIM
from data.preprocess import CLASS_TO_INDEX, INDEX_TO_CLASS

ATTACK_CLASSES = list(CLASS_TO_INDEX.keys())


@dataclass
class ForecastResult:
    predicted_states: np.ndarray
    predicted_states_unscaled: Optional[np.ndarray]
    risk_probabilities: np.ndarray
    category_probabilities: np.ndarray
    predicted_categories: np.ndarray
    category_names: List[List[str]]
    horizon_steps: List[int]
    horizon_seconds: List[int]

    @property
    def composite_threat_probabilities(self) -> np.ndarray:
        benign_prob = self.category_probabilities[..., 0]
        return np.clip(1.0 - benign_prob, 0.0, 1.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "predicted_states": self.predicted_states.tolist(),
            "predicted_states_unscaled": (
                self.predicted_states_unscaled.tolist()
                if self.predicted_states_unscaled is not None else None
            ),
            "risk_probabilities": self.risk_probabilities.tolist(),
            "composite_threat_probabilities": self.composite_threat_probabilities.tolist(),
            "category_probabilities": self.category_probabilities.tolist(),
            "predicted_categories": self.predicted_categories.tolist(),
            "category_names": self.category_names,
            "horizon_steps": self.horizon_steps,
            "horizon_seconds": self.horizon_seconds,
        }

    def get_sequence(self, index: int) -> Dict[str, Any]:
        if self.predicted_states.ndim == 2:
            idx = 0
            states = self.predicted_states
            unscaled = self.predicted_states_unscaled
            risks = self.risk_probabilities
            cat_probs = self.category_probabilities
            pred_cats = self.predicted_categories
            cat_names = self.category_names[0] if self.category_names else []
        else:
            idx = index
            states = self.predicted_states[idx]
            unscaled = self.predicted_states_unscaled[idx] if self.predicted_states_unscaled is not None else None
            risks = self.risk_probabilities[idx]
            cat_probs = self.category_probabilities[idx]
            pred_cats = self.predicted_categories[idx]
            cat_names = self.category_names[idx] if idx < len(self.category_names) else []

        return {
            "sample_index": idx,
            "horizon_seconds": self.horizon_seconds,
            "risk_probabilities": risks,
            "predicted_categories": pred_cats,
            "category_names": cat_names,
            "category_probabilities": cat_probs,
            "predicted_states": states,
            "predicted_states_unscaled": unscaled,
        }


class RecursiveForecaster:
    def __init__(
        self,
        model: Union[NetworkWorldModel, str, os.PathLike],
        scaler: Optional[Union[NetworkStateScaler, str, os.PathLike]] = None,
        device: str = "cpu",
        window_seconds: int = 30
    ) -> None:
        self.device = torch.device(device)
        self.window_seconds = window_seconds

        if isinstance(model, (str, os.PathLike)):
            checkpoint_path = str(model)
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Model checkpoint not found at: {checkpoint_path}")
            
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            model_kwargs = checkpoint.get("model_kwargs", {})
            self.model = NetworkWorldModel(
                state_dim=model_kwargs.get("state_dim", STATE_VECTOR_DIM),
                hidden_dim=model_kwargs.get("hidden_dim", 128),
                num_lstm_layers=model_kwargs.get("num_lstm_layers", 2),
                num_classes=model_kwargs.get("num_classes", 8),
                dropout=model_kwargs.get("dropout", 0.20),
            )
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model = model

        self.model.to(self.device)
        self.model.eval()

        if isinstance(scaler, (str, os.PathLike)):
            scaler_path = str(scaler)
            if not os.path.exists(scaler_path):
                raise FileNotFoundError(f"Scaler file not found at: {scaler_path}")
            if hasattr(NetworkStateScaler, "load"):
                self.scaler = NetworkStateScaler.load(scaler_path)
            else:
                import pickle
                with open(scaler_path, "rb") as f:
                    self.scaler = pickle.load(f)
        else:
            self.scaler = scaler

    def rollout(
        self,
        sequences: Union[np.ndarray, torch.Tensor],
        K: int = 5,
        unscale: bool = True,
        batch_size: int = 64
    ) -> ForecastResult:
        if K < 1:
            raise ValueError(f"Forecast horizon K must be >= 1, got {K}")

        is_single_sample = False
        if isinstance(sequences, torch.Tensor):
            input_tensor = sequences.float()
        else:
            input_tensor = torch.from_numpy(np.asarray(sequences, dtype=np.float32))

        if input_tensor.ndim == 2:
            is_single_sample = True
            input_tensor = input_tensor.unsqueeze(0)

        if input_tensor.ndim != 3:
            raise ValueError(
                f"Expected input with 2 or 3 dimensions [m, D] or [B, m, D], got shape {input_tensor.shape}"
            )

        num_samples, seq_len, state_dim = input_tensor.shape
        if state_dim != self.model.state_dim:
            raise ValueError(
                f"Input state_dim ({state_dim}) does not match model state_dim ({self.model.state_dim})"
            )

        all_pred_states: List[np.ndarray] = []
        all_risk_probs: List[np.ndarray] = []
        all_cat_probs: List[np.ndarray] = []

        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_window = input_tensor[start_idx:end_idx].clone().to(self.device)

            batch_states: List[torch.Tensor] = []
            batch_risks: List[torch.Tensor] = []
            batch_cats: List[torch.Tensor] = []

            for step in range(K):
                with torch.no_grad():
                    state_next, risk_logit, cat_logits = self.model(batch_window)
                    risk_prob = torch.sigmoid(risk_logit)
                    cat_prob = torch.softmax(cat_logits, dim=-1)

                batch_states.append(state_next.unsqueeze(1))
                batch_risks.append(risk_prob)
                batch_cats.append(cat_prob.unsqueeze(1))

                next_state_slice = state_next.unsqueeze(1)
                batch_window = torch.cat([batch_window[:, 1:, :], next_state_slice], dim=1)

            stacked_states = torch.cat(batch_states, dim=1).cpu().numpy()
            stacked_risks = torch.cat(batch_risks, dim=1).cpu().numpy()
            stacked_cats = torch.cat(batch_cats, dim=1).cpu().numpy()

            all_pred_states.append(stacked_states)
            all_risk_probs.append(stacked_risks)
            all_cat_probs.append(stacked_cats)

        final_states = np.concatenate(all_pred_states, axis=0)
        final_risks = np.concatenate(all_risk_probs, axis=0)
        final_cats = np.concatenate(all_cat_probs, axis=0)
        final_pred_cats = np.argmax(final_cats, axis=-1)

        category_names: List[List[str]] = []
        for i in range(num_samples):
            sample_names = [INDEX_TO_CLASS.get(int(c), "Unknown") for c in final_pred_cats[i]]
            category_names.append(sample_names)

        unscaled_states: Optional[np.ndarray] = None
        if unscale and self.scaler is not None:
            flat_states = final_states.reshape(-1, state_dim)
            unscaled_flat = self.scaler.inverse_transform(flat_states)
            unscaled_states = unscaled_flat.reshape(num_samples, K, state_dim)

        horizon_steps = list(range(1, K + 1))
        horizon_seconds = [step * self.window_seconds for step in horizon_steps]

        if is_single_sample:
            return ForecastResult(
                predicted_states=final_states[0],
                predicted_states_unscaled=unscaled_states[0] if unscaled_states is not None else None,
                risk_probabilities=final_risks[0],
                category_probabilities=final_cats[0],
                predicted_categories=final_pred_cats[0],
                category_names=category_names,
                horizon_steps=horizon_steps,
                horizon_seconds=horizon_seconds,
            )

        return ForecastResult(
            predicted_states=final_states,
            predicted_states_unscaled=unscaled_states,
            risk_probabilities=final_risks,
            category_probabilities=final_cats,
            predicted_categories=final_pred_cats,
            category_names=category_names,
            horizon_steps=horizon_steps,
            horizon_seconds=horizon_seconds,
        )


def evaluate_rollout_metrics(
    forecast_result: ForecastResult,
    y_risk_true: np.ndarray,
    y_class_true: np.ndarray,
    y_state_true: Optional[np.ndarray] = None,
    risk_threshold: float = 0.5
) -> Dict[str, Any]:
    pred_risks = forecast_result.risk_probabilities
    pred_cats = forecast_result.predicted_categories
    pred_states = forecast_result.predicted_states

    if pred_risks.ndim == 1:
        pred_risks = pred_risks[np.newaxis, :]
        pred_cats = pred_cats[np.newaxis, :]
        if pred_states.ndim == 2:
            pred_states = pred_states[np.newaxis, :, :]

    N, K = pred_risks.shape
    results: Dict[str, Any] = {
        "num_samples": N,
        "horizons": {},
        "summary": {}
    }

    horizon_f1s = []
    horizon_recalls = []
    horizon_accuracies = []
    horizon_state_mses = []

    for k_idx in range(K):
        step_num = k_idx + 1
        seconds = forecast_result.horizon_seconds[k_idx]
        key = f"step_{step_num}_{seconds}s"

        k_risk_probs = pred_risks[:, k_idx]
        k_risk_pred = (k_risk_probs >= risk_threshold).astype(int)
        k_risk_true = y_risk_true[:, k_idx].astype(int)

        k_cat_pred = pred_cats[:, k_idx]
        k_cat_true = y_class_true[:, k_idx].astype(int)

        tp = int(np.sum((k_risk_pred == 1) & (k_risk_true == 1)))
        fp = int(np.sum((k_risk_pred == 1) & (k_risk_true == 0)))
        tn = int(np.sum((k_risk_pred == 0) & (k_risk_true == 0)))
        fn = int(np.sum((k_risk_pred == 0) & (k_risk_true == 1)))

        acc = (tp + tn) / max(N, 1)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = (2 * prec * rec) / max(prec + rec, 1e-8)
        fpr = fp / max(fp + tn, 1)

        cat_correct = int(np.sum(k_cat_pred == k_cat_true))
        cat_acc = cat_correct / max(N, 1)

        state_mse = None
        state_mae = None
        if y_state_true is not None:
            if y_state_true.ndim == 2 and k_idx == 0:
                diff = pred_states[:, 0, :] - y_state_true
                state_mse = float(np.mean(diff ** 2))
                state_mae = float(np.mean(np.abs(diff)))
            elif y_state_true.ndim == 3 and y_state_true.shape[1] > k_idx:
                diff = pred_states[:, k_idx, :] - y_state_true[:, k_idx, :]
                state_mse = float(np.mean(diff ** 2))
                state_mae = float(np.mean(np.abs(diff)))

        results["horizons"][key] = {
            "step": step_num,
            "horizon_seconds": seconds,
            "risk_accuracy": float(acc),
            "risk_precision": float(prec),
            "risk_recall": float(rec),
            "risk_f1": float(f1),
            "false_positive_rate": float(fpr),
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
            "category_accuracy": float(cat_acc),
            "state_mse": state_mse,
            "state_mae": state_mae,
        }

        horizon_f1s.append(float(f1))
        horizon_recalls.append(float(rec))
        horizon_accuracies.append(float(acc))
        if state_mse is not None:
            horizon_state_mses.append(state_mse)

    results["summary"] = {
        "mean_risk_f1": float(np.mean(horizon_f1s)),
        "risk_f1_by_horizon": horizon_f1s,
        "risk_recall_by_horizon": horizon_recalls,
        "risk_accuracy_by_horizon": horizon_accuracies,
        "state_mse_by_horizon": horizon_state_mses if horizon_state_mses else None,
    }

    return results
