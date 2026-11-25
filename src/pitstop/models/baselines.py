"""Implement train-prevalence, logistic-regression, and unweighted LightGBM baselines."""

import json
import pickle

import lightgbm as lgb  # import before torch (segfault on one Mac)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

FILES = {"constant": "model.json", "logreg": "model.pkl", "lightgbm": "model.txt"}


def _average_precision(y_true, y_score):
    return "average_precision", average_precision_score(y_true, y_score), True


def fit(config, x_train, y_train, x_val, y_val):
    """Fit one baseline on train rows; val rows only drive LightGBM early stopping."""
    kind, seed, params = config["model"], config["seed"], config["parameters"]
    if kind == "constant":
        return float(y_train.mean()), []
    if kind == "logreg":
        return LogisticRegression(**params, random_state=seed).fit(x_train, y_train), []
    stop = config["early_stopping"]
    model = lgb.LGBMClassifier(**params, metric="None", random_state=seed)
    model.fit(
        x_train,
        y_train,
        eval_X=x_val,
        eval_y=y_val,
        eval_metric=_average_precision,
        callbacks=[lgb.early_stopping(stop["rounds"], verbose=False)],
    )
    scores = model.evals_result_["valid_0"]["average_precision"]
    history = [{"iteration": i + 1, "val_average_precision": s} for i, s in enumerate(scores)]
    history.append({"best_iteration": model.best_iteration_})
    return model.booster_, history


def save(model, kind, run_dir):
    path = run_dir / FILES[kind]
    if kind == "constant":
        path.write_text(json.dumps({"train_prevalence": model}) + "\n")
    elif kind == "logreg":
        path.write_bytes(pickle.dumps(model))  # trusted project artifact only
    else:
        model.save_model(path, num_iteration=model.best_iteration)


def load(kind, run_dir):
    path = run_dir / FILES[kind]
    if kind == "constant":
        return json.loads(path.read_text())["train_prevalence"]
    if kind == "logreg":
        return pickle.loads(path.read_bytes())
    return lgb.Booster(model_file=str(path))


def score(model, x):
    """Return probabilities in [0, 1]."""
    if isinstance(model, float):
        return [model] * len(x)
    if isinstance(model, LogisticRegression):
        return model.predict_proba(x)[:, 1]
    return model.predict(x)
