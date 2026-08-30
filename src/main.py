import io
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Network World Model Inference Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
        self.dynamics_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
        )
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

STAGE_MAP = {
    0: "Benign (Normal)",
    1: "Reconnaissance",
    2: "Initial Access",
    3: "Lateral Movement",
    4: "C2 / Exfiltration",
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AttentionWorldModel(input_dim=6, hidden_dim=64, num_classes=5).to(device)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../models/network_world_model.pth")

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
else:
    model.eval()

@app.post("/upload-csv")
async def forecast_from_csv(file: UploadFile = File(...), k_steps: int = 6):
    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid CSV file format.")

    if len(df) < 12:
        raise HTTPException(status_code=400, detail="CSV must contain at least 12 rows (60s of telemetry).")
    
    # Extract the last 12 rows and first 6 feature columns
    seq = df.iloc[-12:, :6].values.astype(np.float32)
    curr_seq = torch.tensor(seq).unsqueeze(0).to(device)
    
    trajectories = []
    final_attn_weights = None

    with torch.no_grad():
        for step in range(1, k_steps + 1):
            pred_state, pred_logits, attn_weights = model(curr_seq)
            probs = torch.softmax(pred_logits, dim=-1).cpu().numpy()[0]
            pred_stage_idx = int(np.argmax(probs))

            trajectories.append({
                "step": step,
                "time_offset": f"t + {step * 5:02d}s",
                "predicted_stage": STAGE_MAP.get(pred_stage_idx, "Unknown"),
                "stage_id": pred_stage_idx,
                "confidence": float(probs[pred_stage_idx])
            })

            if step == 1:
                final_attn_weights = attn_weights.cpu().numpy().squeeze().tolist()

            next_state_reshaped = pred_state.unsqueeze(1)
            curr_seq = torch.cat([curr_seq[:, 1:, :], next_state_reshaped], dim=1)

    return {
        "status": "success",
        "rollout_horizon_seconds": k_steps * 5,
        "trajectories": trajectories,
        "explainability": {
            "temporal_attention_weights": final_attn_weights
        },
        "latest_features": seq[-1].tolist()
    }