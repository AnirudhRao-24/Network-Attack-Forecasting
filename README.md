# Network World Model: Predictive Cyber Defense & Attack Trajectory Forecaster

> **Problem Statement ID:** SIH PS 26153 | **Domain:** Artificial Intelligence & Cybersecurity (NTRO)  
> **Core Innovation:** Causal Network World Modeling via State-Transition Dynamics ($P(S_{t+1} \mid S_t)$) for Proactive Infiltration Forecasting.

---

## 1. System Overview
Traditional Intrusion Detection Systems (IDS) evaluate isolated network packets or flows in static, memoryless isolation, alerting defenders only *after* a breach has occurred. 

This project implements an **Attention-Augmented Network World Model**. Instead of operating as a static classifier, it learns the underlying environment state-transition probability distribution across time-windowed network telemetry. Using an autoregressive $K$-step forward simulation, the engine anticipates attacker progression and maps future states to **MITRE ATT&CK** stages before a compromise is finalized.

---

## 2. Repository File Structure
```text
Network-Attack-Forecasting/
├── data/                      # Telemetry sample generators & test arrays
│   ├── __init__.py
│   └── generate_sample.py     # Script to generate sample CSVs and numpy sequences
├── models/                    # Serialized neural weights
│   └── network_world_model.pth# Pretrained Attention World Model weights
├── notebooks/                 # Research and training pipeline notebooks
│   └── Network_Attack_Forecasting.ipynb
├── src/                       # Production backend microservice
│   ├── __init__.py            # Package declaration
│   ├── model.py               # Attention-Augmented LSTM World Model architecture
│   ├── preprocessor.py        # Dynamic raw/processed CSV parsing pipeline
│   └── main.py                # FastAPI REST endpoints & inference execution
├── .gitignore                 # Git ignore rules for virtualenvs and caches
├── README.md                  # Comprehensive setup & project documentation
├── requirements.txt           # Production Python dependencies
├── index.html                 # Full-screen brutalist web operations dashboard
├── script.js                  # Frontend state machine & backend API connector
└── style.css                  # Responsive high-contrast terminal UI styles

3. Prerequisites & System Requirements
    Python: Version 3.10 or higher

    Package Manager: pip

    Hardware: CPU or NVIDIA GPU with CUDA support (optional for local acceleration)

4. Local Installation & Setup
  Step 1: Clone the Repository

    git clone [https://github.com/](https://github.com/)<your-username>/Network-Attack-Forecasting.git
    cd Network-Attack-Forecasting

  Step 2: Configure a Virtual Environment

    # Create a clean virtual environment
    python -m venv venv
    # Activate the environment
    # On macOS / Linux:
    source venv/bin/activate
    # On Windows (Command Prompt / PowerShell):
    venv\Scripts\activate

  Step 3: Install Dependencies

    Install all required libraries specified in requirements.txt:
    pip install --upgrade pip
    pip install -r requirements.txt

5. Generating Sample Telemetry Data

  To test your ingestion pipeline without needing multi-gigabyte raw datasets immediately, use the built-in data generator to create a sample test file:
  python data/generate_sample.py

  This generates data/demo_telemetry.csv containing synthetic flow and packet anomalies mimicking port reconnaissance activity.

6. Running the System Locally
  Step 1: Launch the Backend Inference Engine (FastAPI)

    Start the local API server using Uvicorn:
    uvicorn src.main:app --host 127.0.0.1 --port 8000 --reload

    *The API will run locally at http://127.0.0.1:8000.

    *Interactive Swagger documentation is available at http://127.0.0.1:8000/docs.

  Step 2: Launch the Frontend Operations Dashboard

    You can view the dashboard by opening index.html directly in your browser, or by serving it locally via Python:
    python -m http.server 3000
    Navigate to http://localhost:3000 in your web browser.

7. Using the DashboardSelect Forecast Horizon ($K$-Steps):
    1.Choose how many future time windows you want to simulate (e.g., 6 steps = 30 seconds ahead).
    2.Upload Telemetry File: Click the file input and upload either a preprocessed 6-column matrix or a raw CIC-IDS traffic log (.csv).
    3.Execute Trajectory Rollout: Click EXECUTE TRAJECTORY ROLLOUT. The engine will stream the future progression phases, display the temporal attention breakdown, and update the real-time telemetry indicators.

8. Cloud Deployment Guide
  Backend Deployment (Render)
  Push your code to GitHub (ensure models/network_world_model.pth and src/ are included).

  Log into Render, create a new Web Service, and link your GitHub repository.

  Configure the service using these exact parameters:

  Environment: Python 3

  Build Command: pip install -r requirements.txt
  
  Start Command: python -m uvicorn src.main:app --host 0.0.0.0 --port $PORT

  Deploy the service and copy your public Render URL (e.g., https://network-attack-forecasting.onrender.com).
