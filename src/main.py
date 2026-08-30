import io
import os
import numpy as np
import pandas as pd
import scipy.stats
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
    def __init__(self, input_dim=6, hidden_dim=64, num_classes=5, num_layers=2):
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

def calculate_entropy(series):
    counts = series.value_counts()
    return scipy.stats.entropy(counts)

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
        df = pd.read_csv(io.BytesIO(contents), low_memory=False)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid CSV file format.")

    # Standardize column headers for reliable parsing
    df.columns = df.columns.str.strip().str.lower()
    
    processed_cols = ['flow_count', 'byte_volume', 'syn_ratio', 'rst_ratio', 'port_entropy', 'iat_mean']
    raw_cols = ['timestamp', 'dst port', 'totlen fwd pkts', 'syn flag cnt', 'rst flag cnt', 'flow iat mean']

    # Route 1: Data is already processed into 6 columns
    if all(col in df.columns for col in processed_cols):
        seq_df = df[processed_cols]

    # Route 2: Data is raw CIC-IDS2018 format and needs on-the-fly aggregation
    elif all(col in df.columns for col in raw_cols):
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(subset=['timestamp'], inplace=True)
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df.dropna(subset=['timestamp'], inplace=True)
        df.sort_values('timestamp', inplace=True)
        df.set_index('timestamp', inplace=True)

        seq_df = df.resample('5S').agg(
            flow_count=('dst port', 'count'),
            byte_volume=('totlen fwd pkts', 'sum'),
            syn_ratio=('syn flag cnt', 'mean'),
            rst_ratio=('rst flag cnt', 'mean'),
            port_entropy=('dst port', calculate_entropy),
            iat_mean=('flow iat mean', 'mean')
        )
        seq_df.ffill(inplace=True)
        seq_df.dropna(inplace=True)
        
    # Route 3: Unknown file structure
    else:
        raise HTTPException(status_code=400, detail="Unrecognized CSV format. Please upload raw CIC-IDS traffic logs or a pre-processed 6-column matrix.")

    if len(seq_df) < 12:
        raise HTTPException(status_code=400, detail="Not enough traffic. The file must contain at least 60 seconds of telemetry.")
    
    # Extract the final 12 rows as the model input tensor
    seq = seq_df.iloc[-12:].values.astype(np.float32)
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
