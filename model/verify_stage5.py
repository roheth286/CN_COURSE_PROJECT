"""
Verification and Unit Test Suite for Stage 5: Network World Model

This script tests:
1. Model instantiation, parameter registration, and architecture structure.
2. Forward pass output tensor dimensions (State: [B, D], Risk: [B, 1], Cat: [B, 8]).
3. Multi-Task loss computation across all 3 objectives (MSE, BCE, CrossEntropy).
4. Backward pass and gradient backpropagation through the LSTM backbone and all 3 heads.
5. High-level inference methods (predict_risk_probability, predict_attack_category).
6. Checkpoint saving and loading weight integrity roundtrip.
"""

import os
import sys
import tempfile
import torch
import numpy as np
from model.network_world_model import NetworkWorldModel, MultiTaskWorldModelLoss


def run_unit_tests() -> None:
    """Run automated unit tests for Stage 5 World Model."""
    print("=" * 60)
    print("Running Stage 5 Network World Model Unit Tests...")
    print("=" * 60)

    # Hyperparameters
    batch_size = 16
    seq_len = 10
    state_dim = 36
    hidden_dim = 128
    num_classes = 8

    # Test 1: Model instantiation
    model = NetworkWorldModel(
        state_dim=state_dim,
        hidden_dim=hidden_dim,
        num_lstm_layers=2,
        num_classes=num_classes,
        dropout=0.2
    )
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total_params > 0 and total_params == trainable_params, "Parameter registration error."
    print(f"[PASS] Test 1: Model instantiated successfully ({total_params:,} parameters).")

    # Test 2: Forward pass output shapes
    dummy_x = torch.randn(batch_size, seq_len, state_dim)
    pred_state, pred_risk_logits, pred_cat_logits = model(dummy_x)

    assert pred_state.shape == (batch_size, state_dim), (
        f"State prediction shape mismatch: {pred_state.shape} vs {(batch_size, state_dim)}"
    )
    assert pred_risk_logits.shape == (batch_size, 1), (
        f"Risk logits shape mismatch: {pred_risk_logits.shape} vs {(batch_size, 1)}"
    )
    assert pred_cat_logits.shape == (batch_size, num_classes), (
        f"Category logits shape mismatch: {pred_cat_logits.shape} vs {(batch_size, num_classes)}"
    )
    print(f"[PASS] Test 2: Forward pass output tensor dimensions verified.")

    # Test 3: Multi-task loss computation
    criterion = MultiTaskWorldModelLoss(lambda_state=1.0, lambda_risk=1.0, lambda_cat=1.0)
    
    target_state = torch.randn(batch_size, state_dim)
    target_risk = torch.randint(0, 2, (batch_size,)).float()
    target_class = torch.randint(0, num_classes, (batch_size,)).long()

    loss, loss_components = criterion(
        pred_state=pred_state,
        pred_risk_logits=pred_risk_logits,
        pred_cat_logits=pred_cat_logits,
        target_state=target_state,
        target_risk=target_risk,
        target_class=target_class
    )
    assert loss.item() > 0, "Loss must be positive."
    assert "state_loss" in loss_components and "risk_loss" in loss_components and "cat_loss" in loss_components
    print(f"[PASS] Test 3: Multi-task loss calculated (Total: {loss.item():.4f}, State: {loss_components['state_loss']:.4f}, Risk: {loss_components['risk_loss']:.4f}, Cat: {loss_components['cat_loss']:.4f}).")

    # Test 4: Backward pass and gradient flow through all heads and LSTM
    model.zero_grad()
    loss.backward()

    # Verify gradients in LSTM backbone
    assert model.lstm.weight_ih_l0.grad is not None, "Gradients not flowing to LSTM layer 0."
    assert model.lstm.weight_hh_l0.grad is not None, "Gradients not flowing to LSTM recurrent weights."
    assert model.lstm.weight_ih_l1.grad is not None, "Gradients not flowing to LSTM layer 1."

    # Verify gradients in all 3 heads
    assert model.state_decoder_head[-1].weight.grad is not None, "Gradients missing in State Decoder head."
    assert model.risk_head[-1].weight.grad is not None, "Gradients missing in Risk head."
    assert model.category_head[-1].weight.grad is not None, "Gradients missing in Category head."
    print("[PASS] Test 4: Gradients verified across LSTM backbone and all three output heads.")

    # Test 5: Inference helper methods
    probs = model.predict_risk_probability(dummy_x)
    assert probs.shape == (batch_size,), f"Expected risk shape {(batch_size,)}, got {probs.shape}"
    assert (probs >= 0.0).all() and (probs <= 1.0).all(), "Risk probabilities must be in [0, 1]."

    pred_classes, cat_probs = model.predict_attack_category(dummy_x)
    assert pred_classes.shape == (batch_size,), f"Expected classes shape {(batch_size,)}, got {pred_classes.shape}"
    assert cat_probs.shape == (batch_size, num_classes)
    assert torch.allclose(cat_probs.sum(dim=-1), torch.ones(batch_size), atol=1e-5), "Softmax probs must sum to 1.0."
    print("[PASS] Test 5: Probability calibration and category prediction methods verified.")

    # Test 6: Checkpoint roundtrip save and load
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": {
                "state_dim": state_dim,
                "hidden_dim": hidden_dim,
                "num_classes": num_classes
            }
        }, tmp_path)

        new_model = NetworkWorldModel(state_dim=state_dim, hidden_dim=hidden_dim, num_classes=num_classes)
        checkpoint = torch.load(tmp_path, weights_only=True)
        new_model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        new_model.eval()

        with torch.no_grad():
            orig_out = model(dummy_x)[0]
            new_out = new_model(dummy_x)[0]
            assert torch.allclose(orig_out, new_out, atol=1e-6), "Checkpoint reload output mismatch."
        print("[PASS] Test 6: Checkpoint save and reload integrity verified.")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    print("\n" + "=" * 60)
    print("All Stage 5 World Model Unit Tests Passed Successfully!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_unit_tests()
