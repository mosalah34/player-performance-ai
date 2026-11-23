# Player Performance AI

A small machine-learning pipeline that predicts a football player's **next-match rating (0-10)** from recent form and match statistics, and labels the player's form as *rising*, *stable* or *declining*.

Built by Eng. Mohamed Salah (Elzehary) as part of an IoT and AI engineering portfolio.

## How it works

1. Match features: goals, assists, passes, pass accuracy, distance covered, sprints, minutes played, and the average rating of the last 5 matches.
2. Two regressors are trained and compared on a held-out split (Ridge regression and Gradient Boosting). The one with the lower mean absolute error is saved.
3. The prediction is compared with the player's last-5 average to label the form trend.

## Quick start

```bash
pip install -r requirements.txt
python player_ai.py demo
```

Example output on the included synthetic sample data:

```
ridge:             MAE 0.28, R2 0.68
gradient_boosting: MAE 0.30, R2 0.64
selected: ridge
```

Predict for one match:

```bash
python player_ai.py predict --goals 1 --assists 0 --passes 48 --pass_acc 86 \
    --distance_km 10.4 --sprints 24 --minutes 90 --last5_avg 7.1
```

## Use your own data

Provide a CSV with the columns `goals, assists, passes, pass_acc, distance_km, sprints, minutes, last5_avg, rating` and run:

```bash
python player_ai.py train --csv matches.csv
```

## Important note on data

The `demo` command generates **synthetic** data so the pipeline runs end to end. The scores above describe that synthetic data only, not real players. For real predictions, train on a real match log.

## Roadmap

- Position-specific models (defender, midfielder, forward)
- Wearable sensor input (heart rate, GPS) from an ESP32 tracker
- A simple web dashboard

## License

MIT
