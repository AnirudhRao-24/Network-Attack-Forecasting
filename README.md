# Network World Model: Predictive Cyber Defense & Attack Trajectory Forecaster

> **Problem Statement ID:** SIH PS 26153 | **Domain:** Artificial Intelligence & Cybersecurity (NTRO)  
> **Core Innovation:** Causal Network World Modeling via State-Transition Dynamics for Proactive Infiltration Forecasting.

## 1. System Overview
Traditional Intrusion Detection Systems (IDS) treat network flows in isolation, classifying traffic snapshots into binary benign/malicious labels. This ignores the temporal kill-chain progression of an attack (e.g., reconnaissance scanning -> initial exploit burst -> lateral movement -> exfiltration). 

This system implements an **Attention-Augmented Network World Model**. Instead of acting as a static classifier, it learns the underlying environment state-transition probability distribution across time-windowed network telemetry. By autoregressively rolling out forward trajectories $K$ steps into the future, the engine anticipates attacker progression and maps future states to **MITRE ATT&CK** stages before a compromise is finalized.

## 2. Key Capabilities
* **Dual-Tier Traffic Ingestion:** Ingests CSV telemetry combining aggregate flow statistics (flow volume, mean inter-arrival time) with packet-level signatures (TCP SYN/RST ratios, destination port entropy).
* **Joint State-Dynamics Learning:** Jointly optimizes dynamics regression (MSE) and discrete MITRE ATT&CK phase forecasting (CrossEntropy) using a Dual-Head LSTM.
* **Autoregressive Rollout:** Projects network states up to 30+ seconds into the future without relying on external oracle features.
* **Explainable AI (XAI):** Built-in Temporal Attention isolates the exact historical time window driving the prediction, mapped directly to the UI.

## 3. Architecture & Tech Stack
* **Frontend:** Vanilla HTML, CSS (Brutalist UI), and JavaScript. Hosted on GitHub Pages.
* **Backend:** FastAPI microservice handling CORS and REST endpoints. Hosted on Render.
* **Machine Learning:** PyTorch (LSTM + Temporal Attention), Pandas, NumPy.

## 4. Directory Layout
```text
Network-Attack-Forecasting/
├── data/                      # Telemetry data
│   └── demo_sequence.npy      # Pre-formatted 12-step lookback sample
├── models/                    # Serialized neural weights
│   └── network_world_model.pth# Pretrained Attention World Model weights
├── notebooks/                 # Research, training, and verification pipelines
│   └── Network_Attack_Forecasting.ipynb
├── src/                       # Production backend microservice
│   └── main.py                # FastAPI inference service & autoregressive engine
├── .gitignore
├── README.md
├── requirements.txt           # Production dependencies
├── index.html                 # Real-time threat projection dashboard
├── script.js                  # Frontend state machine & API handler
└── style.css                  # High-contrast, scannable terminal UI
