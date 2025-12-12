from autogluon.timeseries import TimeSeriesDataFrame
from autogluon.timeseries.metrics.abstract import TimeSeriesScorer
from autogluon.timeseries.metrics import MAPE
import itertools
import pandas as pd
import numpy as np


class MAPE_GroupFairnessScorer(TimeSeriesScorer):
    optimized_by_median = True

    def __init__(self, alpha=0.5, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    def compute_metric(
        self,
        data_future: TimeSeriesDataFrame,
        predictions: TimeSeriesDataFrame,
        target: str = "target",
        **kwargs,
    ) -> float:
        # code copied from MAPE TimeSeriesScorer
        y_true, y_pred = self._get_point_forecast_score_inputs(
            data_future, predictions, target=target
        )
        y_true, y_pred = y_true.to_numpy(), y_pred.to_numpy()
        errors = (np.abs(y_true - y_pred) / np.abs(y_true)).reshape(
            [-1, self.prediction_length]
        )

        if self.horizon_weight is not None:
            errors *= self.horizon_weight

        static_df = data_future.static_features.reset_index()
        mape_cols = []
        for (
            c
        ) in (
            static_df.columns
        ):  # calculate for every column the group fairness and average it at the end
            mape_vals = []
            for val in static_df[
                c
            ].unique():  # calculate for each distinct value the mean mape
                idx = static_df.loc[static_df[c] == val].index
                mape_vals.append(self._safemean(errors[idx]))

            pairwise_diffs = (
                []
            )  # calculate the differences between the mean mape of the groups
            for a, b in itertools.combinations(mape_vals, 2):
                pairwise_diffs.append(abs(a - b))
            mape_cols.append(self._safemean(np.array(pairwise_diffs)))

        return (1 - self.alpha) * self._safemean(errors) + self.alpha * self._safemean(
            np.array(mape_cols)
        )


def evaluate_predictions(
    test_data, train_data, predictor, predictions, eval_metrics=["RMSE", "MAE", "MAPE"]
):
    """
    For every column in test_data.static_features, compute the requested metrics for each distinct
    value of that column and return a dict mapping column -> DataFrame with per-value metric
    results and summary statistics (std, cv, max_diff, mean_diff).
    """
    static_df = (
        test_data.static_features.reset_index()
    )  # contains item_id + static cols
    ts_df = (
        test_data.reset_index()
    )  # time-series rows with item_id, timestamp, target, ...
    merged = pd.merge(ts_df, static_df, on="item_id", how="left")

    results = {}
    static_cols = [c for c in static_df.columns if c != "item_id"]

    for c in static_cols:
        per_value = {}
        values = static_df[c].dropna().unique()
        for (
            val
        ) in (
            values
        ):  # calculate the performance measure for each subgroup of the column
            item_ids = static_df.loc[static_df[c] == val, "item_id"].unique()
            subset = merged[merged["item_id"].isin(item_ids)].copy()
            static_subset = static_df[
                static_df["item_id"].isin(item_ids)
            ].drop_duplicates()

            ts_subset = TimeSeriesDataFrame.from_data_frame(
                subset,
                id_column="item_id",
                timestamp_column="timestamp",
                static_features_df=static_subset,
            )

            eval_res = predictor.evaluate(ts_subset, metrics=eval_metrics)

            per_value[val] = pd.Series(eval_res)

        all_metrics_vals = pd.DataFrame(per_value)
        # ensure numeric where possible
        all_metrics_vals = all_metrics_vals.apply(pd.to_numeric, errors="coerce")

        # add statistic columns
        stat_cols = ["std", "cv", "max_diff", "mean_diff"]
        for sc in stat_cols:
            all_metrics_vals[sc] = np.nan

        # compute stats across group columns (exclude the stat columns themselves)
        group_cols = [col for col in all_metrics_vals.columns if col not in stat_cols]
        for metric in all_metrics_vals.index:
            s = all_metrics_vals.loc[metric, group_cols].dropna().astype(float)
            if s.empty:
                continue
            std = s.std()
            mean = s.mean()
            max_diff = s.max() - s.min()
            pairwise = [abs(a - b) for a, b in itertools.combinations(s, 2)]
            mean_diff = float(np.mean(pairwise)) if pairwise else 0.0

            all_metrics_vals.at[metric, "std"] = std
            all_metrics_vals.at[metric, "cv"] = std / mean if mean != 0 else np.nan
            all_metrics_vals.at[metric, "max_diff"] = max_diff
            all_metrics_vals.at[metric, "mean_diff"] = mean_diff

        results[c] = all_metrics_vals

        data_future = test_data.loc[~test_data.index.isin(train_data.index)]
        fair_metric = MAPE_GroupFairnessScorer(alpha=1)
        print(
            "Own metric: ",
            fair_metric.compute_metric(
                data_future=data_future,
                predictions=predictions,
            ),
        )
    return results
