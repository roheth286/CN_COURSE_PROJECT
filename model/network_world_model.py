import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional


class NetworkWorldModel(nn.Module):
    def __init__(
        self,
        state_dim: int = 36,
        hidden_dim: int = 128,
        num_lstm_layers: int = 2,
        num_classes: int = 8,
        dropout: float = 0.2
    ):
        super(NetworkWorldModel, self).__init__()
        
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.num_lstm_layers = num_lstm_layers
        self.num_classes = num_classes
        
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0.0
        )
        
        self.state_decoder_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, state_dim)
        )
        
        self.risk_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.category_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        projected_x = self.encoder(x)
        lstm_out, (hidden_last, cell_last) = self.lstm(projected_x)
        final_latent_state = lstm_out[:, -1, :]
        
        pred_next_state = self.state_decoder_head(final_latent_state)
        pred_risk_logits = self.risk_head(final_latent_state)
        pred_category_logits = self.category_head(final_latent_state)
        
        return pred_next_state, pred_risk_logits, pred_category_logits
        
    def get_latent_representation(self, x: torch.Tensor) -> torch.Tensor:
        projected_x = self.encoder(x)
        lstm_out, _ = self.lstm(projected_x)
        return lstm_out[:, -1, :]
        
    def predict_risk_probability(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            _, risk_logits, _ = self.forward(x)
            risk_prob = torch.sigmoid(risk_logits).squeeze(-1)
            return risk_prob
            
    def predict_attack_category(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            _, _, category_logits = self.forward(x)
            category_probs = torch.softmax(category_logits, dim=-1)
            predicted_classes = torch.argmax(category_probs, dim=-1)
            return predicted_classes, category_probs


class MultiTaskWorldModelLoss(nn.Module):
    def __init__(
        self,
        lambda_state: float = 1.0,
        lambda_risk: float = 1.0,
        lambda_cat: float = 1.0,
        class_weights: Optional[torch.Tensor] = None
    ):
        super(MultiTaskWorldModelLoss, self).__init__()
        self.lambda_state = lambda_state
        self.lambda_risk = lambda_risk
        self.lambda_cat = lambda_cat
        self.state_loss_fn = nn.SmoothL1Loss(beta=1.0)
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        
    def forward(
        self,
        pred_state: torch.Tensor,
        pred_risk_logits: torch.Tensor,
        pred_cat_logits: torch.Tensor,
        target_state: torch.Tensor,
        target_risk: torch.Tensor,
        target_class: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        state_loss = self.state_loss_fn(pred_state, target_state)
        state_mse = self.mse_loss(pred_state, target_state)
        
        target_risk_expanded = target_risk.view(-1, 1).float()
        risk_loss = self.bce_loss(pred_risk_logits, target_risk_expanded)
        cat_loss = self.ce_loss(pred_cat_logits, target_class.long())
        
        total_loss = (
            self.lambda_state * state_loss
            + self.lambda_risk * risk_loss
            + self.lambda_cat * cat_loss
        )
        
        loss_components = {
            "total_loss": float(total_loss.item()),
            "state_loss": float(state_loss.item()),
            "state_mse": float(state_mse.item()),
            "risk_loss": float(risk_loss.item()),
            "cat_loss": float(cat_loss.item()),
        }
        
        return total_loss, loss_components
