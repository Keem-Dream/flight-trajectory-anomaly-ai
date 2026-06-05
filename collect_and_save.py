"""
collect_and_save.py - Phase 2, Step 4
Collect trajectories, compute features, and SAVE them to a CSV
so we can train on the data later without re-collecting every time.
"""

import time
import math
import pandas as pd
from pyopensky.rest import REST

POLL_SECONDS = 20
NUM_POLLS = 15          # more polls = longer paths = better data (~5 min)
OUTPUT_FILE = "flight_features.csv"

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlam/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def collect():
    api = REST()
    trajectories = {}
    for poll in range(1, NUM_POLLS + 1):
        print(f"--- Poll {poll} of {NUM_POLLS} ---")
        try:
            df = api.states()
        except TypeError:
            df = None
        if df is not None and not df.empty:
            print(f"    got {len(df)} aircraft")
            for _, row in df.iterrows():
                if pd.isna(row["latitude"]) or pd.isna(row["longitude"]):
                    continue
                trajectories.setdefault(row["icao24"], []).append({
                    "t": float(row["timestamp"].timestamp()),
                    "lat": row["latitude"],
                    "lon": row["longitude"],
                    "alt": row["altitude"],
                })
        if poll < NUM_POLLS:
            time.sleep(POLL_SECONDS)
    return trajectories

def build_feature_rows(trajectories):
    """Turn trajectories into a flat list of feature rows for the CSV."""
    rows = []
    for icao, readings in trajectories.items():
        if len(readings) < 2:
            continue
        for prev, curr in zip(readings, readings[1:]):
            dt = curr["t"] - prev["t"]
            if dt <= 0:
                continue
            dist_km = haversine_km(prev["lat"], prev["lon"], curr["lat"], curr["lon"])
            speed_ms = (dist_km * 1000) / dt
            if pd.isna(curr["alt"]) or pd.isna(prev["alt"]):
                climb_ms = 0.0
            else:
                climb_ms = (curr["alt"] - prev["alt"]) / dt
            rows.append({
                "icao24": icao,
                "dt_s": round(dt, 1),
                "dist_km": round(dist_km, 3),
                "speed_ms": round(speed_ms, 2),
                "climb_ms": round(climb_ms, 2),
                "is_fake": 0,          # all real for now; we add fakes in Step 5
            })
    return rows

def main():
    print("Collecting trajectories (this takes a few minutes)...\n")
    trajectories = collect()

    rows = build_feature_rows(trajectories)
    if not rows:
        print("\nNo usable feature rows collected. Try running again.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\n=== Saved {len(df)} feature rows to {OUTPUT_FILE} ===")
    print(f"Aircraft represented: {df['icao24'].nunique()}\n")
    print("Quick stats on speed (m/s):")
    print(df["speed_ms"].describe())

if __name__ == "__main__":
    main()
