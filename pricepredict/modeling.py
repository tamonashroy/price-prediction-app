"""Train return models and choose between them with FLAML AutoML."""

import logging

import numpy as np
import pandas as pd
from flaml import AutoML, tune
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.ensemble import VotingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pricepredict import config

logger = logging.getLogger(__name__)

SELECTION_FOLDS = 4
SELECTION_FOLD_DAYS = 60


def target_column(horizon):
    """What the models learn: the forward return relative to the median coin."""
    return f"excess_target_{horizon}"


def actual_return_column(horizon):
    """What a holder really earns, including the market's own move."""
    return f"target_{horizon}"


def default_tree_model():
    """Fixed, heavily regularised LightGBM settings that held up in the design tests."""
    return LGBMRegressor(
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=15,
        min_child_samples=200,
        colsample_bytree=0.6,
        subsample=0.7,
        subsample_freq=1,
        reg_lambda=10.0,
        random_state=config.RANDOM_SEED,
        n_jobs=-1,
        verbose=-1,
    )


def make_blend(tree_model=None):
    """
    Average a tree model with a heavily shrunk linear model. Their mistakes differ, and
    the blend ranked coins better than either alone in the design tests.
    """
    ridge = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1000.0))
    return VotingRegressor([("trees", tree_model if tree_model is not None else default_tree_model()),
                            ("ridge", ridge)])


def rank_ic(dates, actual, predicted):
    """
    Average same-day Spearman correlation between forecasts and outcomes.
    This measures what picking coins needs: ordering them correctly each day.
    """
    frame = pd.DataFrame(
        {"date": np.asarray(dates), "actual": np.asarray(actual), "predicted": np.asarray(predicted)}
    ).dropna()
    daily = [
        day["actual"].corr(day["predicted"], method="spearman")
        for _, day in frame.groupby("date")
        if len(day) >= 10
    ]
    daily = [value for value in daily if not np.isnan(value)]
    return float(np.mean(daily)) if daily else 0.0


def training_frame(data, horizon, as_of, window_days=config.TRAIN_WINDOW_DAYS):
    """
    Rows whose outcome was already known on `as_of`. A row dated D has an outcome
    at D + horizon, so anything later would leak the future into training.
    """
    last_known = pd.Timestamp(as_of) - pd.Timedelta(days=horizon)
    first = last_known - pd.Timedelta(days=window_days)
    rows = data[(data["date"] <= last_known) & (data["date"] >= first)]
    return rows[rows[target_column(horizon)].notna()]


def clip_target(target):
    """Limit extreme moves so a few wild coins do not dominate the fit."""
    return target.clip(target.quantile(0.01), target.quantile(0.99))


class WalkForwardSplitter:
    """
    Expanding time folds with a gap, for FLAML's cross-validation. Each fold trains on the
    past and validates on a later block. A single validation block is not enough: in data
    this noisy, a search can always find settings that happen to suit one period.
    """

    def __init__(self, dates, horizon, n_splits=SELECTION_FOLDS, fold_days=SELECTION_FOLD_DAYS):
        self.dates = pd.DatetimeIndex(dates)
        self.horizon = horizon
        self.n_splits = n_splits
        self.fold_days = fold_days
        self.dates_by_fold_size = {}

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X=None, y=None, groups=None):
        last = self.dates.max()
        gap = pd.Timedelta(days=self.horizon)
        for fold in range(self.n_splits, 0, -1):
            valid_end = last - pd.Timedelta(days=self.fold_days * (fold - 1))
            valid_start = valid_end - pd.Timedelta(days=self.fold_days)
            train_index = np.flatnonzero(self.dates <= valid_start - gap)
            valid_index = np.flatnonzero((self.dates > valid_start) & (self.dates <= valid_end))
            # FLAML hands the metric only the fold's rows, so find their dates by fold size.
            self.dates_by_fold_size[len(valid_index)] = self.dates[valid_index]
            yield train_index, valid_index


def select_model(data, feature_columns, as_of, horizon=config.PICK_HORIZON,
                 time_budget=config.AUTOML_TIME_BUDGET_SECONDS, verbose=0):
    """
    Search model families and settings with FLAML, scoring each by average rank IC across
    several walk-forward folds. Search ranges are limited to well-regularised models.

    Returns:
        (configured unfitted estimator, description dict)
    """
    rows = training_frame(data, horizon, as_of, config.AUTOML_TRAIN_WINDOW_DAYS).sort_values("date")
    splitter = WalkForwardSplitter(rows["date"].to_numpy(), horizon)

    def rank_ic_loss(X_val, y_val, estimator, labels, X_train, y_train, *args, **kwargs):
        score = rank_ic(splitter.dates_by_fold_size[len(y_val)], y_val, estimator.predict(X_val))
        return -score, {"rank_ic": score}

    automl = AutoML()
    automl.fit(
        X_train=rows[feature_columns].reset_index(drop=True),
        y_train=clip_target(rows[target_column(horizon)]).reset_index(drop=True),
        task="regression",
        metric=rank_ic_loss,
        eval_method="cv",
        split_type=splitter,
        estimator_list=list(config.AUTOML_ESTIMATORS),
        custom_hp=_regularised_search_space(),
        time_budget=time_budget,
        sample=False,
        seed=config.RANDOM_SEED,
        retrain_full=False,
        verbose=verbose,
    )
    description = {
        "estimator": automl.best_estimator,
        "config": {key: _plain(value) for key, value in automl.best_config.items()},
        "validation_rank_ic": -float(automl.best_loss),
    }
    logger.info("AutoML chose %s (validation rank IC %.4f)", automl.best_estimator, description["validation_rank_ic"])
    return clone(automl.model.estimator), description


def _regularised_search_space():
    """Rule out tiny or wildly flexible models, which only look good by fitting noise."""
    n_estimators = {"domain": tune.lograndint(100, 800), "init_value": 200, "low_cost_init_value": 100}
    min_samples = {"domain": tune.lograndint(50, 1000), "init_value": 200, "low_cost_init_value": 200}
    return {
        "lgbm": {
            "n_estimators": n_estimators,
            "min_child_samples": min_samples,
            "num_leaves": {"domain": tune.lograndint(4, 32), "init_value": 15, "low_cost_init_value": 4},
            "learning_rate": {"domain": tune.loguniform(0.005, 0.1), "init_value": 0.03},
        },
        "xgboost": {
            "n_estimators": n_estimators,
            "min_child_weight": {"domain": tune.loguniform(10, 1000), "init_value": 100},
            "max_leaves": {"domain": tune.lograndint(4, 32), "init_value": 15, "low_cost_init_value": 4},
            "learning_rate": {"domain": tune.loguniform(0.005, 0.1), "init_value": 0.03},
        },
    }


def _plain(value):
    return value.item() if isinstance(value, np.generic) else value


def fit_for_horizon(estimator, data, feature_columns, horizon, as_of):
    """Fit a copy of `estimator` on everything known by `as_of` for one horizon."""
    rows = training_frame(data, horizon, as_of)
    model = clone(estimator)
    model.fit(rows[feature_columns], clip_target(rows[target_column(horizon)]))
    return model


def feature_importance(model, feature_columns, top=15):
    """Largest normalised importances, for the dashboard. Empty if the model has none."""
    trees = model.named_estimators_["trees"] if isinstance(model, VotingRegressor) else model
    values = getattr(trees, "feature_importances_", None)
    if values is None:
        return {}
    series = pd.Series(np.asarray(values, dtype=float), index=feature_columns)
    total = series.sum()
    if total > 0:
        series = series / total
    top_values = series.sort_values(ascending=False).head(top)
    return {name: round(float(value), 4) for name, value in top_values.items()}
