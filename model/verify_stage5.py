import os
import sys
import tempfile

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import numpy as np
from model.network_world_model import NetworkWorldModel, MultiTaskWorldModelLoss


def run_unit_tests() -> None:
    print("=" * 60)
    print("Running Stage 5 Network World Model Unit Tests...")
    print("=" * 60)

    batch_size = 16
    seq_len = 10
    state_dim = 36
    hidden_dim = 128
    num_classes = 8

    model = NetworkWorldModel(
        state_dim=state_dim,
        hidden_dim=hidden_dim,
        num_lstm_layers=2,
        num_classes=num_classes,
        dropout=0.2
    )
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total_params > 0 and total_params == trainable_params
    print(f"[PASS] Test 1: Model instantiated successfully ({total_params:,} parameters).")

    dummy_x = torch.randn(batch_size, seq_len, state_dim)
    pred_state, pred_risk_logits, pred_cat_logits = model(dummy_x)

    assert pred_state.shape == (batch_size, state_dim)
    assert pred_risk_logits.shape == (batch_size, 1)
    assert pred_cat_logits.shape == (batch_size, num_classes)
    print("[PASS] Test 2: Forward pass output tensor dimensions verified.")

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
    assert loss.item() > 0
    assert "state_loss" in loss_components and "risk_loss" in loss_components and "cat_loss" in loss_components
    print(f"[PASS] Test 3: Multi-task loss calculated (Total: {loss.item():.4f}, State: {loss_components['state_loss']:.4f}, Risk: {loss_components['risk_loss']:.4f}, Cat: {loss_components['cat_loss']:.4f}).")

    model.zero_grad()
    loss.backward()

    assert model.lstm.weight_ih_l0.grad is not None
    assert model.lstm.weight_hh_l0.grad is not None
    assert model.lstm.weight_ih_l1.grad is not None

    assert model.state_decoder_head[-1].weight.grad is not None
    assert model.risk_head[-1].weight.grad is not None
    assert model.category_head[-1].weight.grad is not None
    print("[PASS] Test 4: Gradients verified across LSTM backbone and all three output heads.")

    probs = model.predict_risk_probability(dummy_x)
    assert probs.shape == (batch_size,)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()

    pred_classes, cat_probs = model.predict_attack_category(dummy_x)
    assert pred_classes.shape == (batch_size,)
    assert cat_probs.shape == (batch_size, num_classes)
    assert torch.allclose(cat_probs.sum(dim=-1), torch.ones(batch_size), atol=1e-5)
    print("[PASS] Test 5: Probability calibration and category prediction methods verified.")

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
            assert torch.allclose(orig_out, new_out, atol=1e-6)
        print("[PASS] Test 6: Checkpoint save and reload integrity verified.")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    print("\n" + "=" * 60)
    print("All Stage 5 World Model Unit Tests Passed Successfully!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_unit_tests()
