"""
inject_fakes.py - Phase 2, Step 5
Take real flight_features.csv and add labeled fake (spoofed) rows:
a realistic mix of obvious and subtle attacks.
Output: flight_features_labeled.csv
"""

import random
import pandas as pd

INPUT_FILE = "flight_features.csv"
OUTPUT_FILE = "flight_features_labeled.csv"
random.seed(42)  # reproducible fakes: same result every run

def make_fake(kind):
    """Build one fake feature row of a given attack type."""
    icao = "fake" + str(random.randint(1000, 9999))
    dt = random.choice([15, 20, 25])  # similar gaps to real data

    if kind == "teleport_obvious":
        # absurd jump: hundreds of km in one interval -> impossible speed
        dist = random.uniform(150, 400)
        speed = (dist * 1000) / dt
        climb = random.uniform(-5, 5)

    elif kind == "teleport_subtle":
        # only slightly too fast: faster than any airliner, but not absurd
        speed = random.uniform(320, 420)   # real cruise tops out ~260
        dist = (speed * dt) / 1000
        climb = random.uniform(-5, 5)

    elif kind == "climb_obvious":
        # impossible vertical rate
        speed = random.uniform(180, 240)
        dist = (speed * dt) / 1000
        climb = random.uniform(120, 250)   # real climb rarely tops ~25 m/s

    elif kind == "climb_subtle":
        # aggressive but borderline-plausible climb
        speed = random.uniform(180, 240)
        dist = (speed * dt) / 1000
        climb = random.uniform(40, 70)

    return {
        "icao24": icao,
        "dt_s": round(dt, 1),
        "dist_km": round(dist, 3),
        "speed_ms": round(speed, 2),
        "climb_ms": round(climb, 2),
        "is_fake": 1,
    }

def main():
    real = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(real)} real rows from {INPUT_FILE}")

    # roughly 15% as many fakes as real rows, mixed across the four types
    n_fakes = max(8, len(real) // 7)
    kinds = ["teleport_obvious", "teleport_subtle", "climb_obvious", "climb_subtle"]
    fakes = [make_fake(random.choice(kinds)) for _ in range(n_fakes)]
    fake_df = pd.DataFrame(fakes)

    # stack real + fake, then shuffle so fakes aren't all at the bottom
    combined = pd.concat([real, fake_df], ignore_index=True)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
    combined.to_csv(OUTPUT_FILE, index=False)

    print(f"Injected {len(fake_df)} fake rows.")
    print(f"Saved {len(combined)} total rows to {OUTPUT_FILE}\n")
    print("Label breakdown:")
    print(combined["is_fake"].value_counts())
    print("\nSpeed range by label (0=real, 1=fake):")
    print(combined.groupby("is_fake")["speed_ms"].agg(["min", "mean", "max"]))

if __name__ == "__main__":
    main()
