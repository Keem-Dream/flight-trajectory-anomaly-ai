"""
collect_trajectories.py - Phase 1, Step 2
Poll OpenSky every ~20s, group readings by aircraft (icao24),
and build a trajectory (ordered path) for each plane.
"""

import time
import pandas as pd
from pyopensky.rest import REST

POLL_SECONDS = 20      # wait time between snapshots
NUM_POLLS = 10         # how many snapshots to collect (10 x 20s = ~3 min)

def main():
    api = REST()

    # trajectories[icao24] = list of readings, in the order we saw them
    trajectories = {}

    for poll in range(1, NUM_POLLS + 1):
        print(f"\n--- Poll {poll} of {NUM_POLLS} ---")
        try:
            df = api.states()
        except TypeError:
            df = None

        if df is None or df.empty:
            print("No data this round (rate limit?). Waiting and retrying...")
        else:
            print(f"Got {len(df)} aircraft. Recording positions...")
            for _, row in df.iterrows():
                icao = row["icao24"]
                reading = {
                    "timestamp": row["timestamp"],
                    "lat": row["latitude"],
                    "lon": row["longitude"],
                    "alt": row["altitude"],
                    "spd": row["groundspeed"],
                }
                # start a new list for a plane we haven't seen, then append
                trajectories.setdefault(icao, []).append(reading)

        # don't sleep after the final poll
        if poll < NUM_POLLS:
            time.sleep(POLL_SECONDS)

    # --- Summary: which aircraft did we actually track over time? ---
    print("\n\n=== Collection complete ===")
    multi = {k: v for k, v in trajectories.items() if len(v) >= 2}
    print(f"Total aircraft seen: {len(trajectories)}")
    print(f"Aircraft with a real path (2+ readings): {len(multi)}\n")

    # show the 5 longest trajectories
    longest = sorted(multi.items(), key=lambda kv: len(kv[1]), reverse=True)[:5]
    for icao, readings in longest:
        print(f"Aircraft {icao}: {len(readings)} readings")
        for r in readings:
            lat = f"{r['lat']:.3f}" if not pd.isna(r["lat"]) else "-"
            lon = f"{r['lon']:.3f}" if not pd.isna(r["lon"]) else "-"
            alt = f"{r['alt']:.0f}" if not pd.isna(r["alt"]) else "-"
            print(f"    {lat:>9}, {lon:>10}   alt={alt:>6}m")
        print()

if __name__ == "__main__":
    main()
