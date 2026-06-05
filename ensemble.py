"""
ensemble.py - The robust detector
Combines Random Forest (specialist: precise on known attacks) with
Isolation Forest (generalist: catches novel weirdness).
Flags a flight if EITHER model catches it.
"""
import random, pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler

FEATURES = ["dt_s","dist_km","speed_ms","climb_ms"]

def make_truly_novel():
    """An 8th attack type NEITHER model trained on:
       normal speed & climb, but distance impossible for the time."""
    dt = random.choice([15,20,25])
    speed = random.uniform(200,250)        # totally normal speed
    climb = random.uniform(-5,5)           # totally normal climb
    dist = random.uniform(20,40)           # but distance says it moved way too far
    return {"dt_s":round(dt,1),"dist_km":round(dist,3),
            "speed_ms":round(speed,2),"climb_ms":round(climb,2)}

def main():
    data = pd.read_csv("flight_features_labeled_v2.csv")

    # Specialist: Random Forest (supervised)
    rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    rf.fit(data[FEATURES], data["is_fake"])

    # Generalist: Isolation Forest (unsupervised) - learns normal from REAL flights only
    real = data[data["is_fake"]==0]
    scaler = StandardScaler()
    iso = IsolationForest(contamination=0.05, random_state=42, n_estimators=200)
    iso.fit(scaler.fit_transform(real[FEATURES]))

    # Generate a TRULY novel attack neither model has seen
    random.seed(777)
    novel = pd.DataFrame([make_truly_novel() for _ in range(500)])

    rf_pred  = rf.predict(novel[FEATURES])
    iso_pred = (iso.predict(scaler.transform(novel[FEATURES])) == -1).astype(int)
    either   = ((rf_pred==1) | (iso_pred==1)).astype(int)

    print("Against a TRULY NOVEL 8th attack type (neither model trained on it):\n")
    print(f"  Random Forest alone:    {int(rf_pred.sum())}/500  ({rf_pred.sum()/500:.0%})")
    print(f"  Isolation Forest alone: {int(iso_pred.sum())}/500  ({iso_pred.sum()/500:.0%})")
    print(f"  ENSEMBLE (either one):  {int(either.sum())}/500  ({either.sum()/500:.0%})")

if __name__ == "__main__":
    main()
