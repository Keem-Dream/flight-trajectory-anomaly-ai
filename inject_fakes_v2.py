"""
inject_fakes_v2.py - Stronger, varied fakes
Includes the original attacks AND the novel ones that beat the model:
descents, slow drifts, erratic. Output: flight_features_labeled_v2.csv
"""

import random
import pandas as pd

random.seed(42)
INPUT_FILE = "flight_features.csv"          # the original REAL data
OUTPUT_FILE = "flight_features_labeled_v2.csv"

def make_fake():
    dt = random.choice([15, 20, 25])
    kind = random.choice([
        "teleport_obvious", "teleport_subtle", "climb_obvious", "climb_subtle",
        "slow_drift", "descent_attack", "erratic"     # the NEW ones
    ])

    if kind == "teleport_obvious":
        dist = random.uniform(150, 400); speed = (dist*1000)/dt; climb = random.uniform(-5,5)
    elif kind == "teleport_subtle":
        speed = random.uniform(320,420); dist = (speed*dt)/1000; climb = random.uniform(-5,5)
    elif kind == "climb_obvious":
        speed = random.uniform(180,240); dist = (speed*dt)/1000; climb = random.uniform(120,250)
    elif kind == "climb_subtle":
        speed = random.uniform(180,240); dist = (speed*dt)/1000; climb = random.uniform(40,70)
    elif kind == "slow_drift":
        speed = random.uniform(270,310); dist = (speed*dt)/1000; climb = random.uniform(-3,3)
    elif kind == "descent_attack":
        speed = random.uniform(180,240); dist = (speed*dt)/1000; climb = random.uniform(-120,-80)
    else:  # erratic
        speed = random.uniform(250,290); dist = (speed*dt)/1000; climb = random.uniform(30,38)

    return {"icao24": "fake"+str(random.randint(1000,9999)),
            "dt_s": round(dt,1), "dist_km": round(dist,3),
            "speed_ms": round(speed,2), "climb_ms": round(climb,2), "is_fake": 1}

def main():
    real = pd.read_csv(INPUT_FILE)
    n_fakes = max(20, len(real)//5)
    fakes = pd.DataFrame([make_fake() for _ in range(n_fakes)])
    combined = pd.concat([real, fakes], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    combined.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(combined)} rows ({len(fakes)} fakes, 7 attack types) to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
