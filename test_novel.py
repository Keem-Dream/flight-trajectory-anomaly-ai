"""
test_novel.py - Stress test
Train on the original fakes, then test against NEW attack types
the model has never seen. Reveals whether it generalized or memorized.
"""

import random
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

random.seed(99)  # different seed = different fakes than training
INPUT_FILE = "flight_features_labeled.csv"
FEATURES = ["dt_s", "dist_km", "speed_ms", "climb_ms"]

def make_novel_fake():
    """Attack types DIFFERENT from the original injector."""
    dt = random.choice([15, 20, 25])
    kind = random.choice(["slow_drift", "descent_attack", "erratic"])

    if kind == "slow_drift":
        # barely-too-fast: just over real cruise, much subtler than before
        speed = random.uniform(270, 310)
        dist = (speed * dt) / 1000
        climb = random.uniform(-3, 3)
    elif kind == "descent_attack":
        # impossible DESCENT (negative climb) - we only trained on positive climbs!
        speed = random.uniform(180, 240)
        dist = (speed * dt) / 1000
        climb = random.uniform(-120, -80)
    else:  # erratic
        speed = random.uniform(250, 290)
        dist = (speed * dt) / 1000
        climb = random.uniform(30, 38)

    return {"dt_s": round(dt,1), "dist_km": round(dist,3),
            "speed_ms": round(speed,2), "climb_ms": round(climb,2)}

def main():
    # Train on ALL the original data (the model's full knowledge)
    data = pd.read_csv(INPUT_FILE)
    model = RandomForestClassifier(n_estimators=200, random_state=42,
                                   class_weight="balanced")
    model.fit(data[FEATURES], data["is_fake"])

    # Generate 500 NOVEL fakes the model has never seen
    novel = pd.DataFrame([make_novel_fake() for _ in range(500)])
    preds = model.predict(novel[FEATURES])

    caught = int(preds.sum())
    print(f"Generated 500 NOVEL fakes (attack types not in training).")
    print(f"Model caught: {caught} / 500  ({caught/500:.0%})\n")

    missed = novel[preds == 0]
    print(f"MISSED {len(missed)} novel fakes. A sample:")
    print(missed.head(15).to_string(index=False))

if __name__ == "__main__":
    main()
