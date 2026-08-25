import joblib
import numpy as np
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from model import NetworkWorldModel

app = FastAPI(title="NTRO World Model Forecaster API")

# Enable CORS for Firebase frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FEATURE_NAMES = [
    "flow_count", "bytes_in", "bytes_out", "pkts_in", "pkts_out",
    "syn_flag_ratio", "ack_flag_ratio", "rst_flag_ratio", "fin_flag_ratio",
    "flow_duration_mean", "iat_mean", "iat_variance",
    "ttl_mean", "ttl_variance", "tcp_window_mean",
    "dest_port_entropy", "src_port_entropy", "payload_entropy"
]

MITRE_STAGES = [
    "Benign Traffic",
    "Reconnaissance (Scan)",
    "Initial Access (Exploit)",
    "Lateral Movement (Probing)",
    "Exfiltration / C2"
]

device = torch.device("cpu")
model = NetworkWorldModel(feature_dim=len(FEATURE_NAMES)).to(device)
model.load_state_dict(torch.load("world_model.pth", map_location=device))
model.eval()

scaler = joblib.load("scaler.pkl")

class SimulationRequest(BaseModel):
    recon_intensity: float = 0.5
    lateral_activity: float = 0.2
    exfil_volume: float = 0.3
    horizon_k: int = 6

@app.post("/api/forecast")
def forecast(req: SimulationRequest):
    # Construct base sequence of 12 historical windows
    base_seq = np.random.normal(loc=50, scale=10, size=(12, len(FEATURE_NAMES)))
    
    # Inject intensity factors into sequence
    base_seq[:, 5] += req.recon_intensity * 0.5    # SYN ratio
    base_seq[:, 15] += req.recon_intensity * 0.4   # Dest Port entropy
    base_seq[:, 13] += req.lateral_activity * 6.0  # TTL variance
    base_seq[:, 2] += req.exfil_volume * 120000    # Outbound bytes
    
    scaled_seq = scaler.transform(base_seq)
    seq_tensor = torch.tensor(scaled_seq, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        _, stage_logits = model.forward_rollout(seq_tensor, K=req.horizon_k)
        probs = torch.softmax(stage_logits, dim=-1).squeeze(0).numpy()
    
    timeline = []
    for k in range(req.horizon_k):
        stage_idx = int(np.argmax(probs[k]))
        risk = float((1.0 - probs[k][0]) * 100.0)
        timeline.append({
            "step": f"t+{k+1} ({(k+1)*5}s)",
            "risk_percentage": round(risk, 2),
            "predicted_stage": MITRE_STAGES[stage_idx],
            "confidence": round(float(probs[k][stage_idx]) * 100, 2)
        })
        
    return {
        "timeline": timeline,
        "max_risk": max(t["risk_percentage"] for t in timeline),
        "dominant_telemetry": FEATURE_NAMES[int(np.argmax(scaled_seq[-1]))]
    }