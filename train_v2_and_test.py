"""
train_v2_and_test.py
Train on the VARIED v2 fakes, then re-test against novel attacks.
Compare to the old 0% to see if the fix worked.
"""
import random, pandas as pd
from sklearn.ensemble import RandomForestClassifier

FEATURES = ["dt_s","dist_km","speed_ms","climb_ms"]

def make_novel():
    dt = random.choice([15,20,25]); k = random.choice(["slow_drift","descent_attack","erratic"])
    if k=="slow_drift": s=random.uniform(270,310); c=random.uniform(-3,3)
    elif k=="descent_attack": s=random.uniform(180,240); c=random.uniform(-120,-80)
    else: s=random.uniform(250,290); c=random.uniform(30,38)
    return {"dt_s":round(dt,1),"dist_km":round((s*dt)/1000,3),"speed_ms":round(s,2),"climb_ms":round(c,2)}

def main():
    data = pd.read_csv("flight_features_labeled_v2.csv")
    model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    model.fit(data[FEATURES], data["is_fake"])

    random.seed(123)  # different fakes than training
    novel = pd.DataFrame([make_novel() for _ in range(500)])
    caught = int(model.predict(novel[FEATURES]).sum())
    print(f"Novel attacks caught after v2 training: {caught}/500 ({caught/500:.0%})")
    print(f"(Was 0/500 before adding varied fakes)")

if __name__ == "__main__":
    main()
