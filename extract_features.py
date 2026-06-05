"""
extract_features.py - Phase 1, Step 3
Turn raw trajectories into measured features:
distance, implied speed, climb rate, heading change between readings.
"""

import time
import math
import pandas as pd
from pyopensky.rest import REST

POLL_SECONDS = 20
NUM_POLLS = 8

def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance between two lat/lon points, in km."""
    R = 6371.0  # Earth radius in km
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
            for _, row in df.iterrows():
                if pd.isna(row["latitude"]) or pd.isna(row["longitude"]):
                    continue  # skip readings with no position
                trajectories.setdefault(row["icao24"], []).append({
                    "t": float(row["timestamp"].timestamp()),
                    "lat": row["latitude"],
                    "lon": row["longitude"],
                    "alt": row["altitude"],
                })
        if poll < NUM_POLLS:
            time.sleep(POLL_SECONDS)
    return trajectories

def features_for(readings):
    """Compute features between consecutive readings of one aircraft."""
    rows = []
    for prev, curr in zip(readings, readings[1:]):
        dt = curr["t"] - prev["t"]
        if dt <= 0:
            continue
        dist_km = haversine_km(prev["lat"], prev["lon"], curr["lat"], curr["lon"])
        speed_ms = (dist_km * 1000) / dt
        climb_ms = ((curr["alt"] - prev["alt"]) / dt
                    if not pd.isna(curr["alt"]) and not pd.isna(prev["alt"]) else float("nan"))
        rows.append({
            "dt_s": round(dt, 1),
            "dist_km": round(dist_km, 2),
            "speed_ms": round(speed_ms, 1),
            "climb_ms": round(climb_ms, 1) if not pd.isna(climb_ms) else None,
        })
    return rows

def main():
    print("Collecting trajectories (this takes a few minutes)...\n")
    trajectories = collect()

    multi = {k: v for k, v in trajectories.items() if len(v) >= 2}
    print(f"\nTracked {len(multi)} aircraft with usable paths.\n")

    longest = sorted(multi.items(), key=lambda kv: len(kv[1]), reverse=True)[:5]
    for icao, readings in longest:
        feats = features_for(readings)
        if not feats:
            continue
        print(f"Aircraft {icao}  ({len(readings)} readings):")
        print(f"  {'dt(s)':>6} {'dist(km)':>9} {'speed(m/s)':>11} {'climb(m/s)':>11}")
        for f in feats:
            climb = f"{f['climb_ms']}" if f["climb_ms"] is not None else "-"
            print(f"  {f['dt_s']:>6} {f['dist_km']:>9} {f['speed_ms']:>11} {climb:>11}")
        print()

if __name__ == "__main__":
    main()
