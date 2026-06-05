"""
train_detector.py - Phase 2, Step 6
Train an Isolation Forest to flag anomalous (likely spoofed) flights,
then check how many of our known fakes it caught.
"""

import pandas as pd
from sklearn.ensemble import IsolationForest

INPUT_FILE = "flight_features_labeled.csv"
FEATURES = ["dt_s", "dist_km", "speed_ms", "climb_ms"]

def main():
    data = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(data)} rows "
          f"({int(data['is_fake'].sum())} fake, "
          f"{int((data['is_fake'] == 0).sum())} real)\n")

    X = data[FEATURES]

    frac_fake = data["is_fake"].mean()
    model = IsolationForest(
        contamination=frac_fake,
        random_state=42,
        n_estimators=200,
    )
    model.fit(X)

    preds = model.predict(X)
    data["flagged"] = (preds == -1).astype(int)

    tp = len(data[(data["is_fake"] == 1) & (data["flagged"] == 1)])
    fn = len(data[(data["is_fake"] == 1) & (data["flagged"] == 0)])
    fp = len(data[(data["is_fake"] == 0) & (data["flagged"] == 1)])
    tn = len(data[(data["is_fake"] == 0) & (data["flagged"] == 0)])

    total_fakes = tp + fn
    recall = tp / total_fakes if total_fakes else 0
    precision = tp / (tp + fp) if (tp + fp) else 0

    print("=== Results ===")
    print(f"Caught fakes (true positives):   {tp}")
    print(f"Missed fakes (false negatives):  {fn}")
    print(f"False alarms (false positives):  {fp}")
    print(f"Correct real   (true negatives): {tn}\n")
    print(f"Recall    (% of fakes caught):      {recall:.0%}")
    print(f"Precision (% of flags that were fake): {precision:.0%}\n")

    print("Fakes the model MISSED (the hard ones):")
    missed = data[(data["is_fake"] == 1) & (data["flagged"] == 0)]
    if missed.empty:
        print("  none - caught them all!")
    else:
        print(missed[["icao24"] + FEATURES].to_string(index=False))

if __name__ == "__main__":
    main()
