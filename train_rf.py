"""
train_rf.py - Phase 2.5
Supervised Random Forest. Learns what fake looks like from labeled examples,
then is tested on data it has NEVER seen.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

INPUT_FILE = "flight_features_labeled.csv"
FEATURES = ["dt_s", "dist_km", "speed_ms", "climb_ms"]

def main():
    data = pd.read_csv(INPUT_FILE)
    X = data[FEATURES]
    y = data["is_fake"]

    # SPLIT: train on 70%, test on the 30% the model never sees.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows (held out)\n")

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    print("=== Results on UNSEEN test data ===")
    print(confusion_matrix(y_test, preds))
    print()
    print(classification_report(y_test, preds, target_names=["real", "fake"]))

    print("Feature importance:")
    for feat, imp in sorted(zip(FEATURES, model.feature_importances_),
                            key=lambda x: -x[1]):
        print(f"  {feat:>10}: {imp:.2f}")

if __name__ == "__main__":
    main()
