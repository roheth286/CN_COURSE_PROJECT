"""
NetForecaster - World Model Training Pipeline (Stage 5)

This script:
1. Loads the sliding sequence datasets from Datasets/sequences/.
2. Computes class-balanced inverse frequency weights for multi-class supervision.
3. Instantiates the NetworkWorldModel and MultiTaskWorldModelLoss.
4. Trains the model using AdamW, gradient clipping, and ReduceLROnPlateau scheduling.
5. Evaluates on validation data every epoch (MSE, Binary F1, Category Accuracy).
6. Automatically saves the best model checkpoint to model/checkpoints/best_world_model.pt.
7. Exports full epoch history to model/checkpoints/training_history.json.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Dict, Tuple, List, Optional
from features.sequence_dataset import NetworkSequenceDataset
from model.network_world_model import NetworkWorldModel, MultiTaskWorldModelLoss
from data.preprocess import INDEX_TO_CLASS


def compute_class_weights(train_classes: np.ndarray, num_classes: int = 8, max_weight: float = 15.0) -> torch.Tensor:
    """
    Compute inverse-frequency class weights to balance the 8-class CrossEntropy loss.
    Minority attacks receive higher weight to prevent the dominant BENIGN class from overwhelming gradients.
    """
    unique, counts = np.unique(train_classes, return_counts=True)
    class_count_dict = dict(zip(unique, counts))
    
    total_samples = len(train_classes)
    weights = []
    
    for c_idx in range(num_classes):
        cnt = class_count_dict.get(c_idx, 0)
        if cnt > 0:
            weight = total_samples / (float(num_classes) * cnt)
            weight = min(weight, max_weight)
        else:
            weight = 1.0
        weights.append(weight)
        
    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    # Normalize so mean weight is 1.0
    weights_tensor = weights_tensor / weights_tensor.mean()
    return weights_tensor


def evaluate_model(
    model: NetworkWorldModel,
    data_loader: DataLoader,
    criterion: MultiTaskWorldModelLoss,
    device: torch.device
) -> Dict[str, float]:
    """Evaluate model performance across all 3 multi-task heads on a given DataLoader."""
    model.eval()
    
    total_loss = 0.0
    state_loss_total = 0.0
    risk_loss_total = 0.0
    cat_loss_total = 0.0
    
    all_pred_risks = []
    all_true_risks = []
    all_pred_classes = []
    all_true_classes = []
    
    with torch.no_grad():
        for x, y_state, y_binary, y_class in data_loader:
            x = x.to(device)
            y_state = y_state.to(device)
            target_risk = y_binary[:, 0].to(device)
            target_class = y_class[:, 0].to(device)
            
            pred_state, pred_risk_logits, pred_cat_logits = model(x)
            
            batch_loss, comps = criterion(
                pred_state=pred_state,
                pred_risk_logits=pred_risk_logits,
                pred_cat_logits=pred_cat_logits,
                target_state=y_state,
                target_risk=target_risk,
                target_class=target_class
            )
            
            total_loss += comps["total_loss"] * len(x)
            state_loss_total += comps["state_loss"] * len(x)
            risk_loss_total += comps["risk_loss"] * len(x)
            cat_loss_total += comps["cat_loss"] * len(x)
            
            # Risk metrics (threshold = 0.5)
            probs = torch.sigmoid(pred_risk_logits).squeeze(-1).cpu().numpy()
            all_pred_risks.extend(probs)
            all_true_risks.extend(target_risk.cpu().numpy())
            
            # Category metrics
            preds = torch.argmax(pred_cat_logits, dim=-1).cpu().numpy()
            all_pred_classes.extend(preds)
            all_true_classes.extend(target_class.cpu().numpy())
            
    num_samples = len(all_true_risks)
    avg_total_loss = total_loss / num_samples
    avg_state_mse = state_loss_total / num_samples
    avg_risk_loss = risk_loss_total / num_samples
    avg_cat_loss = cat_loss_total / num_samples
    
    # Binary Classification Metrics
    all_pred_binary = (np.array(all_pred_risks) >= 0.5).astype(int)
    all_true_binary = np.array(all_true_risks).astype(int)
    
    tp = int(((all_pred_binary == 1) & (all_true_binary == 1)).sum())
    fp = int(((all_pred_binary == 1) & (all_true_binary == 0)).sum())
    tn = int(((all_pred_binary == 0) & (all_true_binary == 0)).sum())
    fn = int(((all_pred_binary == 0) & (all_true_binary == 1)).sum())
    
    accuracy = (tp + tn) / max(num_samples, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = (2 * precision * recall) / max(precision + recall, 1e-7)
    
    # Multi-Class Accuracy
    correct_classes = (np.array(all_pred_classes) == np.array(all_true_classes)).sum()
    cat_accuracy = correct_classes / max(num_samples, 1)
    
    return {
        "val_loss": avg_total_loss,
        "state_mse": avg_state_mse,
        "risk_loss": avg_risk_loss,
        "cat_loss": avg_cat_loss,
        "risk_accuracy": accuracy,
        "risk_precision": precision,
        "risk_recall": recall,
        "risk_f1": f1,
        "cat_accuracy": cat_accuracy,
    }


def train_network_world_model(
    data_dir: str = "Datasets/sequences",
    checkpoint_dir: str = "model/checkpoints",
    epochs: int = 25,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    hidden_dim: int = 128,
    num_lstm_layers: int = 2,
    dropout: float = 0.2,
    patience: int = 7
) -> Dict:
    """Execute complete training loop for the Network World Model."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using compute device: {device}")
    
    # Load dataset archives
    train_npz = np.load(os.path.join(data_dir, "train_sequences.npz"))
    val_npz = np.load(os.path.join(data_dir, "validation_sequences.npz"))
    
    train_dataset = NetworkSequenceDataset(
        x=train_npz["x"],
        y_state=train_npz["y_state"],
        y_binary=train_npz["y_binary"],
        y_class=train_npz["y_class"]
    )
    val_dataset = NetworkSequenceDataset(
        x=val_npz["x"],
        y_state=val_npz["y_state"],
        y_binary=val_npz["y_binary"],
        y_class=val_npz["y_class"]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    state_dim = train_npz["x"].shape[2]  # 36
    seq_len = train_npz["x"].shape[1]    # 10
    num_classes = 8
    
    print("=" * 80)
    print("STAGE 5: Training Network-State World Model")
    print(f"Train Sequences: {len(train_dataset):,} | Val Sequences: {len(val_dataset):,}")
    print(f"Architecture: Input({seq_len}, {state_dim}) -> LSTM({hidden_dim}, layers={num_lstm_layers}) -> 3 Heads")
    print("=" * 80)
    
    # Instantiate model
    model = NetworkWorldModel(
        state_dim=state_dim,
        hidden_dim=hidden_dim,
        num_lstm_layers=num_lstm_layers,
        num_classes=num_classes,
        dropout=dropout
    ).to(device)
    
    # Compute class weights for loss balancing
    train_classes = train_npz["y_class"][:, 0]
    class_weights = compute_class_weights(train_classes, num_classes=num_classes).to(device)
    
    # Loss, Optimizer, Scheduler
    criterion = MultiTaskWorldModelLoss(
        lambda_state=1.0,
        lambda_risk=1.5,
        lambda_cat=1.0,
        class_weights=class_weights
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-5
    )
    
    history: Dict[str, List[float]] = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_state_mse": [],
        "val_risk_f1": [],
        "val_risk_acc": [],
        "val_cat_acc": [],
        "lr": []
    }
    
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_checkpoint_path = os.path.join(checkpoint_dir, "best_world_model.pt")
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_accum = 0.0
        train_state_loss_accum = 0.0
        train_risk_loss_accum = 0.0
        train_cat_loss_accum = 0.0
        
        for x, y_state, y_binary, y_class in train_loader:
            x = x.to(device)
            y_state = y_state.to(device)
            target_risk = y_binary[:, 0].to(device)
            target_class = y_class[:, 0].to(device)
            
            optimizer.zero_grad()
            
            pred_state, pred_risk_logits, pred_cat_logits = model(x)
            
            loss, comps = criterion(
                pred_state=pred_state,
                pred_risk_logits=pred_risk_logits,
                pred_cat_logits=pred_cat_logits,
                target_state=y_state,
                target_risk=target_risk,
                target_class=target_class
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss_accum += comps["total_loss"] * len(x)
            train_state_loss_accum += comps["state_loss"] * len(x)
            train_risk_loss_accum += comps["risk_loss"] * len(x)
            train_cat_loss_accum += comps["cat_loss"] * len(x)
            
        avg_train_loss = train_loss_accum / len(train_dataset)
        
        # Evaluate on validation split
        val_metrics = evaluate_model(model, val_loader, criterion, device)
        scheduler.step(val_metrics["val_loss"])
        current_lr = optimizer.param_groups[0]["lr"]
        
        # Record history
        history["epoch"].append(epoch)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_metrics["val_loss"])
        history["val_state_mse"].append(val_metrics["state_mse"])
        history["val_risk_f1"].append(val_metrics["risk_f1"])
        history["val_risk_acc"].append(val_metrics["risk_accuracy"])
        history["val_cat_acc"].append(val_metrics["cat_accuracy"])
        history["lr"].append(current_lr)
        
        is_best = val_metrics["val_loss"] < best_val_loss
        best_marker = "[BEST]" if is_best else "      "
        
        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train: {avg_train_loss:.4f} | "
            f"Val: {val_metrics['val_loss']:.4f} | "
            f"State MSE: {val_metrics['state_mse']:.4f} | "
            f"Risk F1: {val_metrics['risk_f1']:.1%} | "
            f"Cat Acc: {val_metrics['cat_accuracy']:.1%} | "
            f"LR: {current_lr:.1e} {best_marker}"
        )
        
        if is_best:
            best_val_loss = val_metrics["val_loss"]
            epochs_no_improve = 0
            # Save checkpoint
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "val_metrics": val_metrics,
                "config": {
                    "state_dim": state_dim,
                    "hidden_dim": hidden_dim,
                    "num_lstm_layers": num_lstm_layers,
                    "num_classes": num_classes,
                    "dropout": dropout
                }
            }, best_checkpoint_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\n[Early Stopping] Triggered after {patience} epochs without validation loss improvement.")
                break
                
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Best Checkpoint Saved: {best_checkpoint_path}")
    print("=" * 80)
    
    # Save training history JSON
    history_path = os.path.join(checkpoint_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)
    print(f"[History] Exported training metrics to: {history_path}\n")
    
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NetForecaster World Model")
    parser.add_argument("--data-dir", type=str, default="Datasets/sequences")
    parser.add_argument("--checkpoint-dir", type=str, default="model/checkpoints")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=7)
    args = parser.parse_args()
    
    train_network_world_model(
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        hidden_dim=args.hidden_dim,
        num_lstm_layers=args.layers,
        dropout=args.dropout,
        patience=args.patience
    )
