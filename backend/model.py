import torch
import torch.nn as nn

class NetworkWorldModel(nn.Module):
    def __init__(self, feature_dim=18, latent_dim=64, num_stages=5):
        super(NetworkWorldModel, self).__init__()
        self.encoder_gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=latent_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        self.state_predictor = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, feature_dim)
        )
        self.stage_classifier = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, num_stages)
        )
        
    def forward(self, x_seq):
        gru_out, _ = self.encoder_gru(x_seq)
        return self.state_predictor(gru_out[:, -1, :])
    
    def forward_rollout(self, x_seq, K=6):
        simulated_states = []
        simulated_stage_logits = []
        curr_seq = x_seq.clone()
        
        for _ in range(K):
            gru_out, _ = self.encoder_gru(curr_seq)
            next_state = self.state_predictor(gru_out[:, -1, :])
            stage_logits = self.stage_classifier(next_state)
            
            simulated_states.append(next_state.unsqueeze(1))
            simulated_stage_logits.append(stage_logits.unsqueeze(1))
            curr_seq = torch.cat([curr_seq[:, 1:, :], next_state.unsqueeze(1)], dim=1)
            
        return torch.cat(simulated_states, dim=1), torch.cat(simulated_stage_logits, dim=1)