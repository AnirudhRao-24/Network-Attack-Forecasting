import os
import numpy as np
import pandas as pd

def generate_sample_files():
    data_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. Generate 20 timesteps of synthetic telemetry (6 columns)
    np.random.seed(42)
    telemetry_data = {
        "flow_count": np.random.normal(50, 10, 20),
        "byte_volume": np.random.normal(50000, 10000, 20),
        "syn_ratio": np.random.uniform(0.05, 0.15, 20),
        "rst_ratio": np.random.uniform(0.01, 0.05, 20),
        "port_entropy": np.random.uniform(0.1, 0.3, 20),
        "iat_mean": np.random.uniform(20.0, 40.0, 20)
    }

    # Inject reconnaissance anomaly in the last 4 windows
    telemetry_data["port_entropy"][-4:] = [0.85, 0.92, 0.96, 0.98]
    telemetry_data["syn_ratio"][-4:] = [0.70, 0.82, 0.88, 0.94]
    telemetry_data["flow_count"][-4:] = [210, 240, 260, 290]

    df = pd.DataFrame(telemetry_data)
    csv_path = os.path.join(data_dir, "demo_telemetry.csv")
    df.to_csv(csv_path, index=False)

    # 2. Save 12-step lookback array
    seq_array = df.iloc[-12:].values.astype(np.float32)
    npy_path = os.path.join(data_dir, "demo_sequence.npy")
    np.save(npy_path, seq_array)

    print(f"Generated: {csv_path}")
    print(f"Generated: {npy_path}")

if __name__ == "__main__":
    generate_sample_files()