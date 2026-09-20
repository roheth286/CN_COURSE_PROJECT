"""
NetForecaster - Network-State World Model Architecture (Stage 5)

This module defines the PyTorch LSTM-based Network-State World Model.
The model follows an Encoder -> LSTM -> 3-Head Multi-Task architecture:
1. Backbone: 2-layer LSTM modeling the temporal evolution of network states.
2. Head 1 (State Decoder): Predicts the next continuous network state vector (S_hat_{t+1} in R^D).
3. Head 2 (Threat Risk Head): Predicts the binary attack probability P(Attack_{t+1}) in [0, 1].
4. Head 3 (Attack Category Head): Predicts the multi-class attack logits across the 8 classes.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional


class NetworkWorldModel(nn.Module):
    """
    Temporal Network-State World Model.
    
    Parameters:
    -----------
    state_dim : int
        Dimensionality of the network state vector S_t (default: 36).
    hidden_dim : int
        Hidden dimension of the LSTM and projection layers (default: 128).
    num_lstm_layers : int
        Number of stacked LSTM layers (default: 2).
    num_classes : int
        Number of attack classes including BENIGN (default: 8).
    dropout : float
        Dropout probability between LSTM layers and MLP heads (default: 0.2).
    """
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
        
        # 1. Feature Projection Encoder (D -> H)
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 2. Temporal LSTM Backbone
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0.0
        )
        
        # 3. Head 1: State Decoder Head (H -> D)
        # Learns the physical state transition dynamics S_{t+1}
        self.state_decoder_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, state_dim)
        )
        
        # 4. Head 2: Binary Threat Risk Head (H -> 1)
        # Predicts probability of an attack occurring in the next window
        self.risk_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 5. Head 3: Multi-Class Attack Category Head (H -> Num_Classes)
        # Classifies the specific attack category among the 8 classes
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
        """
        Forward pass through the World Model.
        
        Parameters:
        -----------
        x : torch.Tensor of shape [batch_size, sequence_len_m, state_dim]
            Sequence of m consecutive historical network state vectors.
            
        Returns:
        --------
        pred_next_state : torch.Tensor of shape [batch_size, state_dim]
            Predicted next normalized network state S_hat_{t+1}.
        pred_risk_logits : torch.Tensor of shape [batch_size, 1]
            Raw logit for binary attack threat in window t+1.
        pred_category_logits : torch.Tensor of shape [batch_size, num_classes]
            Raw logits over the 8 attack classes for window t+1.
        """
        # Step 1: Project input features to hidden dimension
        projected_x = self.encoder(x)  # [batch_size, sequence_len, hidden_dim]
        
        # Step 2: Temporal modeling through LSTM
        lstm_out, (hidden_last, cell_last) = self.lstm(projected_x)
        
        # Extract the hidden representation at the final timestep
        final_latent_state = lstm_out[:, -1, :]  # [batch_size, hidden_dim]
        
        # Step 3: Decode predictions from the latent state
        pred_next_state = self.state_decoder_head(final_latent_state)
        pred_risk_logits = self.risk_head(final_latent_state)
        pred_category_logits = self.category_head(final_latent_state)
        
        return pred_next_state, pred_risk_logits, pred_category_logits
        
    def get_latent_representation(self, x: torch.Tensor) -> torch.Tensor:
        """Extract the final hidden state representation h_t for explainability/analysis."""
        projected_x = self.encoder(x)
        lstm_out, _ = self.lstm(projected_x)
        return lstm_out[:, -1, :]
        
    def predict_risk_probability(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the calibrated attack probability in [0, 1] via Sigmoid."""
        with torch.no_grad():
            _, risk_logits, _ = self.forward(x)
            risk_prob = torch.sigmoid(risk_logits).squeeze(-1)
            return risk_prob
            
    def predict_attack_category(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the predicted attack category index and class probabilities via Softmax."""
        with torch.no_grad():
            _, _, category_logits = self.forward(x)
            category_probs = torch.softmax(category_logits, dim=-1)
            predicted_classes = torch.argmax(category_probs, dim=-1)
            return predicted_classes, category_probs


class MultiTaskWorldModelLoss(nn.Module):
    """
    Multi-Task Loss Function for NetForecaster.
    
    Loss = lambda_state * MSE(S_hat, S)
         + lambda_risk * BCEWithLogits(risk_logits, is_attack)
         + lambda_cat * CrossEntropy(cat_logits, class_index)
    """
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
        """Compute the composite multi-task loss and return component metrics."""
        # 1. State prediction loss (Robust SmoothL1 / Huber for heavy attack bursts)
        state_loss = self.state_loss_fn(pred_state, target_state)
        state_mse = self.mse_loss(pred_state, target_state)
        
        # 2. Binary risk loss (BCEWithLogits)
        target_risk_expanded = target_risk.view(-1, 1).float()
        risk_loss = self.bce_loss(pred_risk_logits, target_risk_expanded)
        
        # 3. Multi-class category loss (CrossEntropy)
        cat_loss = self.ce_loss(pred_cat_logits, target_class.long())
        
        # Total combined loss
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
