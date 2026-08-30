import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super(TemporalAttention, self).__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_outputs: torch.Tensor):
        attn_scores = self.attn(lstm_outputs)
        attn_weights = F.softmax(attn_scores, dim=1)
        context_vector = torch.sum(attn_weights * lstm_outputs, dim=1)
        return context_vector, attn_weights

class AttentionWorldModel(nn.Module):
    def __init__(self, input_dim: int = 6, hidden_dim: int = 64, num_classes: int = 5, num_layers: int = 2):
        super(AttentionWorldModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.2)
        self.attention = TemporalAttention(hidden_dim)
        
        # Dynamics Transition Head: P(S_t+1 | S_t)
        self.dynamics_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
        )
        
        # MITRE ATT&CK Stage Classification Head
        self.stage_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor):
        lstm_out, _ = self.lstm(x)
        context, attn_weights = self.attention(lstm_out)
        pred_next_state = self.dynamics_head(context)
        pred_stage_logits = self.stage_head(context)
        return pred_next_state, pred_stage_logits, attn_weights
