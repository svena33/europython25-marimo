import marimo

__generated_with = "0.14.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    return mo, pd


@app.cell
def _(pd):
    import numpy as np
    import matplotlib.pyplot as plt
    from darts import TimeSeries
    from darts.dataprocessing.transformers import Scaler
    from math import floor

    # Parameters
    start_date = "2025-01-01"
    end_date = "2025-12-31"
    freq = "D"  # daily frequency
    peak_day = 3  # Thursday (0=Mon, ..., 6=Sun)
    amplitude = 10
    noise_std = 1.1

    # Create date range
    date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    num_days = len(date_range)

    # Weekly pattern
    weekly_pattern = np.array(
        [100 * np.cos(2 * np.pi * (i - peak_day) / 7) for i in range(7)]
    )
    weekly_pattern = (weekly_pattern - weekly_pattern.min()) / (
        weekly_pattern.max() - weekly_pattern.min()
    )
    pattern_repeated = np.tile(weekly_pattern, num_days // 7 + 1)[:num_days]

    # Add noise
    np.random.seed(42)
    noise = np.random.normal(0, noise_std, size=num_days)
    values = amplitude * pattern_repeated + noise

    # Initialize event list
    events = [None] * num_days  # or [""] * num_days if preferred

    # Define function to label events
    def add_event(dates, label):
        for d in dates:
            try:
                date = pd.Timestamp(d)
                idx = date_range.get_loc(date)
                values[idx] += np.random.poisson(lam=10)
                events[idx] = label
            except KeyError:
                continue

    # Add existing events
    add_event([f"2025-07-{day:02d}" for day in range(14, 21)], "Europython")
    add_event([f"2025-12-{day:02d}" for day in range(24, 26)], "Christmas")
    add_event(["2025-04-20"], "Easter")
    add_event(["2025-01-01"], "New Year")
    add_event(["2025-05-01"], "Labour Day")
    add_event(["2025-07-05"], "Saints Cyril and Methodius Day")
    add_event(["2025-09-28"], "St. Wenceslas Day")

    # Add 100 random promotion events without overlapping existing events
    # Find indices without events
    available_indices = [i for i, e in enumerate(events) if e is None]

    # Check enough free days available
    assert len(available_indices) >= 100, "Not enough free days for promotions"

    np.random.seed(123)  # reproducibility
    promo_indices = np.random.choice(available_indices, size=100, replace=False)

    for idx in promo_indices:
        events[idx] = "Promotion"
        values[idx] += np.random.poisson(lam=10) 

    # Ensure values stay >= 1
    values = np.maximum(1, values)

    # Scale and convert values to int
    values = np.round(values * 100).astype(int)

    # Create DataFrame with results
    df_series = pd.DataFrame({
        "date": date_range,
        "sales_qty": values,
        "event": events
    })

    print(df_series["event"].value_counts())

    # Optional: plot example
    plt.figure(figsize=(14, 5))
    plt.plot(df_series["date"], df_series["sales_qty"], label="Sales Quantity")
    plt.scatter(df_series.loc[df_series["event"] == "Promotion", "date"], 
                df_series.loc[df_series["event"] == "Promotion", "sales_qty"], 
                color="red", label="Promotion Events", s=10)
    plt.title("Sales with Events (Promotions in Red)")
    plt.xlabel("Date")
    plt.ylabel("Sales Quantity")
    plt.legend()
    plt.tight_layout()
    plt.show()
    return TimeSeries, df_series, plt


@app.cell
def _(df_series):
    df_series
    return


@app.cell
def _(df_series):
    df_series.to_csv('data/trdelnik_sales_2.csv', index=False)
    return


@app.cell
def _():
    import altair as alt
    return (alt,)


@app.cell
def _(alt, df_series, mo):
    chart = mo.ui.altair_chart(
        alt.Chart(df_series)
        .mark_point()
        .encode(  # line encode
            x="date",
            y="sales_qty",
            # color='event'
        )
    )
    return (chart,)


@app.cell
def _(chart, mo):
    mo.vstack([chart, mo.ui.table(chart.value)])
    return


@app.cell
def _(alt, df_series, mo):
    mo.ui.altair_chart(
        alt.Chart(df_series)
        .mark_line()
        .encode(  # line encode
            x="date",
            y="sales_qty",
            # color='event'
        )
    )
    return


@app.cell
def _(df_series):
    df_series["has_event"] = df_series["event"].notna().astype(int)
    return


@app.cell
def _(df_series):
    df_series[df_series["has_event"] > 0].values
    return


@app.cell
def _(TimeSeries, df_series):
    # Step 2: Create a multivariate TimeSeries with both 'sales_qty' and 'has_event'
    multi_series = TimeSeries.from_dataframe(
        df_series, time_col="date", value_cols=["sales_qty", "has_event"]
    )
    multi_series
    return (multi_series,)


@app.cell
def _(multi_series, plt):
    ax = multi_series["sales_qty"].plot(label="Sales Quantity")
    multi_series["has_event"].plot(
        label="Has Event", ax=ax.twinx(), color="orange"
    )

    # Combine legends from both axes
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax.get_legend_handles_labels()
    ax.legend(lines + lines2, labels)

    plt.show()
    return


@app.cell
def _(multi_series):
    from darts.utils.statistics import (
        extract_trend_and_seasonality,
        check_seasonality,
    )

    seasonality, season_period = check_seasonality(multi_series["sales_qty"])
    seasonality, season_period
    return


@app.cell
def _(multi_series):
    multi_series["sales_qty"][50:65].plot()
    return


@app.cell
def _(multi_series, pd):
    train, test = multi_series.split_before(pd.Timestamp("20251201"))
    train["sales_qty"].plot(label="training")
    test["sales_qty"].plot(label="test")
    return test, train


@app.cell
def _(test, train):
    from darts.models import NaiveSeasonal

    naive_model = NaiveSeasonal(K=7)
    naive_model.fit(train["sales_qty"])
    naive_forecast = naive_model.predict(31)

    test["sales_qty"].plot(label="actual")
    naive_forecast.plot(label="naive forecast (K=7)")
    return (naive_forecast,)


@app.cell
def _(naive_forecast, test):
    from darts.metrics import mape

    print(
        f"Mean absolute percentage error for the combined naive seasonal: {mape(test['sales_qty'], naive_forecast):.2f}%."
    )
    return


@app.cell
def _():
    import torch
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping

    optimizer_kwargs = {
        "lr": 1e-3,
    }

    # PyTorch Lightning Trainer arguments
    pl_trainer_kwargs = {
        "gradient_clip_val": 1,
        "max_epochs": 2,
        "accelerator": "cpu",
        "callbacks": [],
    }

    # learning rate scheduler
    lr_scheduler_cls = torch.optim.lr_scheduler.ExponentialLR
    lr_scheduler_kwargs = {
        "gamma": 0.999,
    }
    #
    common_model_args = {
        "input_chunk_length": 7,  # lookback window
        "output_chunk_length": 7,  # forecast/lookahead window
        "optimizer_kwargs": optimizer_kwargs,
        "pl_trainer_kwargs": pl_trainer_kwargs,
        "lr_scheduler_cls": lr_scheduler_cls,
        "lr_scheduler_kwargs": lr_scheduler_kwargs,
        "likelihood": None,  # use a likelihood for probabilistic forecasts
        "save_checkpoints": True,  # checkpoint to retrieve the best performing model state,
        "force_reset": True,
        "batch_size": 256,
        "random_state": 42,
    }
    return (common_model_args,)


@app.cell
def _(common_model_args, test, train):
    from darts.models import TiDEModel

    # predicting atmospheric pressure
    target_train = train["sales_qty"]
    past_cov_train = train["has_event"]
    future_cov_train = train["has_event"]

    target_test = test["sales_qty"]
    past_cov_test = test["has_event"]
    future_cov_test = test["has_event"]

    tide_model = TiDEModel(
        **common_model_args, use_reversible_instance_norm=False, model_name="tide0"
    )
    tide_model.fit(
        target_train,
        past_covariates=past_cov_train,
        future_covariates=future_cov_train,
    )
    return future_cov_test, tide_model


@app.cell
def _(future_cov_test, test, tide_model, train):
    pred = tide_model.predict(
        series=train["sales_qty"].concatenate(test["sales_qty"][:-14]),
        past_covariates=future_cov_test,
        future_covariates=future_cov_test,
        n=10,
    )
    pred.values()
    return (pred,)


@app.cell
def _(pred):
    pred.plot()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
