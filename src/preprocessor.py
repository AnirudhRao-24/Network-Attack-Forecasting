import io
import numpy as np
import pandas as pd
import scipy.stats

PROCESSED_COLS = ['flow_count', 'byte_volume', 'syn_ratio', 'rst_ratio', 'port_entropy', 'iat_mean']
RAW_COLS = ['timestamp', 'dst port', 'totlen fwd pkts', 'syn flag cnt', 'rst flag cnt', 'flow iat mean']

def calculate_entropy(series: pd.Series) -> float:
    counts = series.value_counts()
    return float(scipy.stats.entropy(counts))

def parse_telemetry_csv(file_bytes: bytes) -> np.ndarray:
    """
    Parses either pre-processed 6-column telemetry or raw CIC-IDS flow records,
    aggregating into 5-second state windows and returning the last 12 windows.
    """
    try:
        df = pd.read_csv(io.BytesIO(file_bytes), low_memory=False)
    except Exception as exc:
        raise ValueError("Invalid CSV file format.") from exc

    df.columns = df.columns.str.strip().str.lower()

    # Route 1: Pre-processed 6-column matrix
    if all(col in df.columns for col in PROCESSED_COLS):
        seq_df = df[PROCESSED_COLS].copy()

    # Route 2: Raw flow records requiring on-the-fly 5-second aggregation
    elif all(col in df.columns for col in RAW_COLS):
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(subset=['timestamp'], inplace=True)
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df.dropna(subset=['timestamp'], inplace=True)
        df.sort_values('timestamp', inplace=True)
        df.set_index('timestamp', inplace=True)

        seq_df = df.resample('5s').agg(
            flow_count=('dst port', 'count'),
            byte_volume=('totlen fwd pkts', 'sum'),
            syn_ratio=('syn flag cnt', 'mean'),
            rst_ratio=('rst flag cnt', 'mean'),
            port_entropy=('dst port', calculate_entropy),
            iat_mean=('flow iat mean', 'mean')
        )
        seq_df.ffill(inplace=True)
        seq_df.dropna(inplace=True)
    else:
        raise ValueError("Unrecognized CSV format. Upload standard CIC-IDS flow traffic or pre-aggregated telemetry.")

    if len(seq_df) < 12:
        raise ValueError("Insufficient time history. Telemetry file must contain at least 12 rows (60 seconds).")

    return seq_df.iloc[-12:].values.astype(np.float32)
