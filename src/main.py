import os
import torch
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.model import AttentionWorldModel
from src.preprocessor import parse_telemetry_csv

app = FastAPI(title="Network World Model Inference Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

@app.get("/health")
def health_check():
    return {"status": "online", "model_loaded": os.path.exists(MODEL_PATH)}

@app.post("/upload-csv")
async def forecast_from_csv(file: UploadFile = File(...), k_steps: int = 6):
    contents = await file.read()
    
    try:
        seq = parse_telemetry_csv(contents)
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err))

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
