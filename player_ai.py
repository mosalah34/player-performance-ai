"""Player Performance AI.

Predict a football player's next-match performance rating (0-10) from recent
form and physical/tactical stats, and classify the player's form trend.

Usage:
    python player_ai.py demo                    # generate sample data, train, evaluate
    python player_ai.py train --csv matches.csv # train on your own match log
    python player_ai.py predict --goals 1 --assists 0 --passes 48 \
        --pass_acc 86 --distance_km 10.4 --sprints 24 --minutes 90 --last5_avg 7.1

CSV columns for your own data:
    goals, assists, passes, pass_acc, distance_km, sprints, minutes,
    last5_avg, rating   (rating = the target, 0-10)
"""
import argparse
import json
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import pickle

FEATURES = ["goals", "assists", "passes", "pass_acc", "distance_km",
            "sprints", "minutes", "last5_avg"]
MODEL_PATH = "model.pkl"


def make_sample_data(n=1500, seed=7):
    """Synthetic match log. It is NOT real player data: it exists so the
    pipeline can be run and tested end to end."""
    rng = np.random.default_rng(seed)
    minutes = rng.choice([30, 45, 60, 75, 90], n, p=[.05, .1, .15, .2, .5])
    goals = rng.poisson(0.25 * minutes / 90)
    assists = rng.poisson(0.2 * minutes / 90)
    passes = rng.normal(0.55 * minutes, 8).clip(5)
    pass_acc = rng.normal(82, 7, n).clip(50, 99)
    distance = rng.normal(0.115 * minutes, 0.8).clip(1)
    sprints = rng.normal(0.25 * minutes, 4).clip(0)
    last5 = rng.normal(6.6, 0.6, n).clip(4.5, 9)
    rating = (5.2 + 0.9 * goals + 0.6 * assists + 0.012 * (pass_acc - 80)
              + 0.08 * (distance - 0.115 * minutes) + 0.01 * (sprints - 0.25 * minutes)
              + 0.25 * (last5 - 6.6) + rng.normal(0, 0.35, n))
    df = pd.DataFrame(dict(goals=goals, assists=assists, passes=passes.round(),
                           pass_acc=pass_acc.round(1), distance_km=distance.round(2),
                           sprints=sprints.round(), minutes=minutes,
                           last5_avg=last5.round(2), rating=rating.clip(3, 10).round(2)))
    return df


def train(df):
    X, y = df[FEATURES], df["rating"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)
    candidates = {
        "ridge": Ridge(alpha=1.0),
        "gradient_boosting": GradientBoostingRegressor(random_state=0),
    }
    best, best_mae, report = None, 1e9, {}
    for name, model in candidates.items():
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        mae = mean_absolute_error(y_te, pred)
        report[name] = {"mae": round(mae, 3), "r2": round(r2_score(y_te, pred), 3)}
        if mae < best_mae:
            best, best_mae, best_name = model, mae, name
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(best, f)
    report["selected"] = best_name
    return report


def trend(prediction, last5_avg):
    delta = prediction - last5_avg
    if delta > 0.4:
        return "rising"
    if delta < -0.4:
        return "declining"
    return "stable"


def predict(row):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    x = pd.DataFrame([row])[FEATURES]
    p = float(np.clip(model.predict(x)[0], 0, 10))
    return {"predicted_rating": round(p, 2), "form": trend(p, row["last5_avg"])}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("demo")
    t = sub.add_parser("train")
    t.add_argument("--csv", required=True)
    p = sub.add_parser("predict")
    for f in FEATURES:
        p.add_argument(f"--{f}", type=float, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "demo":
        df = make_sample_data()
        df.to_csv("sample_matches.csv", index=False)
        print(json.dumps(train(df), indent=2))
        print(predict(dict(goals=1, assists=0, passes=48, pass_acc=86,
                           distance_km=10.4, sprints=24, minutes=90, last5_avg=7.1)))
    elif a.cmd == "train":
        print(json.dumps(train(pd.read_csv(a.csv)), indent=2))
    else:
        print(json.dumps(predict({f: getattr(a, f) for f in FEATURES})))


if __name__ == "__main__":
    sys.exit(main())
