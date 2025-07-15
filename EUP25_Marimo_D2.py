import marimo

__generated_with = "0.14.10"
app = marimo.App(width="medium")

with app.setup:
    # Initialization code that runs before all other cells
    SEED = 20250715

    import numpy as np
    import marimo as mo
    import pandas as pd
    import altair as alt

    from darts import TimeSeries
    from darts.models import LightGBMModel

    np.random.seed(20250715)


@app.cell
def _():
    mo.md(
        r"""
    <div style="text-align: center;">
        <h2>📈 Trdelník Sales Forecasting</h2>
            <p><em>A notebook showcasing various Marimo features by <a href="https://svenarends.com">Sven Arends</a>.</em></p>
    </em></p>
    </div>
    """
    )
    return


@app.cell
def _():
    mo.hstack(
        [
            mo.image(
                src="data/Trdelník (Chimney Cake).jpg",
                rounded=True,
                width=250,
                height=200,
            )
        ],
        justify="center",
    )
    return


@app.cell
def _():
    mo.md(
        """
    <div style="text-align: center;">
    <h3>Step 1: Load and Inspect the Data</h3>
    <p>We start by loading the sales data into a DataFrame. Note that we parse the <code>date</code> column as datetime objects for easier time-series handling.
    We also create a new binary column <code>has_event</code> indicating whether a special event was happening on that date.</p>
    </div>
    """
    )
    return


@app.cell
def _():
    trdelnik_sales = pd.read_csv("data/trdelnik_sales.csv", parse_dates=["date"])
    trdelnik_sales["has_event"] = trdelnik_sales["event"].notna().astype(int)

    trdelnik_sales
    return (trdelnik_sales,)


@app.cell
def _():
    mo.md(
        r"""
    <div style="text-align: center;">
    <h3>Step 2: Exploratory Data Analysis (EDA): Special Days Overview</h3>
    <p>Let's get a sense of which special events are present in the dataset and how frequently they occur.</p>
    </div>
    """
    )
    return


@app.cell
def _(trdelnik_sales):
    _chart = (
        alt.Chart(trdelnik_sales)
        .mark_bar()
        .transform_aggregate(count="count()", groupby=["event"])
        .transform_window(
            rank="rank()",
            sort=[
                alt.SortField("count", order="descending"),
                alt.SortField("event", order="ascending"),
            ],
        )
        .transform_filter(alt.datum.rank <= 10)
        .encode(
            y=alt.Y(
                "event:N",
                sort="-x",
                axis=alt.Axis(title=None),
            ),
            x=alt.X("count:Q", title="Number of records"),
            tooltip=[
                alt.Tooltip("event:N", title="Event"),
                alt.Tooltip("count:Q", format=",.0f", title="Number of Records"),
            ],
        )
        .properties(
            width="container",
            height=400,
            title={
                "text": "Special Events with Frequency",
                "anchor": "middle",
            },
        )
        .configure_view(stroke=None)
        .configure_axis(grid=False)
    )
    mo.ui.altair_chart(_chart)
    return


@app.cell
def _():
    mo.md(
        r"""
    <div style="text-align: center;">
    <h3>Step 3: EDA — Investigating Sales Data</h3>
    <p>Let's take a closer look at the sales quantity time series to identify trends, seasonality, or outliers.</p>
    </div>
    """
    )
    return


@app.cell
def _():
    mo.md(
        r"""
    <div style="text-align: center;">
    <h4>Step 3.1: Exploring Outliers</h4>
    <p>Visualizing sales over time as discrete points helps detect unusual spikes or drops that may require special attention.</p>
    </div>
    """
    )
    return


@app.cell
def _(trdelnik_sales):
    sales_point_chart = (
        alt.Chart(trdelnik_sales)
        .mark_point()
        .encode(
            x=alt.X(
                "date",
                axis=alt.Axis(labelAngle=45, format="%b %Y", title="Month"),
                title="Date",
            ),
            y=alt.Y("sales_qty", axis=alt.Axis(title="Sales Quantity")),
            tooltip=[
                alt.Tooltip("date", title="Date", format="%Y-%m-%d"),
                alt.Tooltip("sales_qty", title="Sales Qty"),
                alt.Tooltip("event", title="Event"),
            ],
        )
        .properties(
            title={
                "text": "Daily Trdelník Sales",
                "anchor": "middle",
            },
        )
    )
    sales_point_chart_ui = mo.ui.altair_chart(sales_point_chart)
    return (sales_point_chart_ui,)


@app.cell
def _(sales_point_chart_ui):
    mo.vstack([sales_point_chart_ui, mo.ui.table(sales_point_chart_ui.value)])
    return


@app.cell
def _():
    mo.md(
        """
    <div style="text-align: center;">
    <strong>Complete Data Table</strong><br />
    Below is the full dataset which you can further explore or export if needed.
    </div>
    """
    )
    return


@app.cell
def _(trdelnik_sales):
    mo.ui.dataframe(trdelnik_sales)
    return


@app.cell
def _():
    mo.md(
        r"""
    <div style="text-align: center;">
    <h3>Step 4: Seasonality Exploration</h3>
    <p>Time series data often contains seasonal cycles. Let's examine if sales of Trdelník have recurring patterns.</p>
    </div>
    """
    )
    return


@app.cell
def _(trdelnik_sales):
    line_chart = (
        alt.Chart(trdelnik_sales)
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "date",
                axis=alt.Axis(labelAngle=45, format="%b %Y", title="Month"),
            ),
            y=alt.Y("sales_qty", axis=alt.Axis(title="Sales Quantity")),
            tooltip=[
                alt.Tooltip("date", format="%Y-%m-%d", title="Date"),
                alt.Tooltip("sales_qty", title="Sales Qty"),
            ],
        )
        .properties(
            title={"text": "Trdelník Sales Over Time", "anchor": "middle"},
        )
    )
    mo.ui.altair_chart(line_chart)
    return


@app.cell
def _():
    mo.md(
        """
    <div style="text-align: center;">
    The chart above suggests some repeating patterns over time, hinting at seasonality.<br/>
    Next, we'll use the auto correlation statistical test from the <a href="https://unit8co.github.io/darts/index.html">darts</a> library to confirm this hypothesis.
    </div>
    """
    )
    return


@app.cell
def _(trdelnik_sales):
    from darts.utils.statistics import check_seasonality

    trdelnik_sales_ts = TimeSeries.from_dataframe(
        trdelnik_sales, time_col="date", value_cols=["sales_qty", "has_event"]
    )

    _seasonal_pattern_bool, _seasonal_period = check_seasonality(
        trdelnik_sales_ts["sales_qty"]
    )

    mo.md(
        f"""
        <div style="text-align: center;">
        <h4>4.1 Seasonality Check Results</h3>
        <ul style="list-style:none; padding-left:0;">
          <li>🔍 <b>Is there a seasonal pattern?</b> <span style="color:green; font-weight:bold;">{_seasonal_pattern_bool}</span></li>
          <li>📅 <b>Length of seasonal period (days):</b> <span style="font-weight:bold;">{_seasonal_period}</span></li>
        </ul>
        </div>
        """
    )
    return (trdelnik_sales_ts,)


@app.cell
def _():
    mo.md(
        """
    <div style="text-align: center;">
    <h3>Step 5: Prepare Train-Test Split</h3>
    </div>
    """
    )
    return


@app.cell
def _(trdelnik_sales_ts):
    train, test = trdelnik_sales_ts.split_before(pd.Timestamp("20251201"))
    return test, train


@app.cell
def _():
    mo.md(
        """
    <div style="text-align: center;">
    <h3>Step 6: Baseline Model: Naive Seasonal Forecasting</h3>
    A simple benchmark model that repeats sales from the last season (weekly).
    </div>
    """
    )
    return


@app.cell
def _(train):
    from darts.models import NaiveSeasonal

    naive_model = NaiveSeasonal(K=7)  # weekly seasonality
    naive_model.fit(train["sales_qty"])
    return (naive_model,)


@app.cell
def _(naive_model, test):
    naive_forecast = naive_model.predict(31)

    test["sales_qty"].plot(label="Actual Sales")
    naive_forecast.plot(label="Baseline Forecast")
    return (naive_forecast,)


@app.cell
def _(naive_forecast, test):
    from darts.metrics import mape

    baseline_mape_error = mape(test["sales_qty"], naive_forecast)

    mo.md(
        f"""
        <div style="text-align: center;">
        <h4>6.1 Baseline Model Evaluation</h3>
        The Mean Absolute Percentage Error (MAPE) on the test set is approximately <b>{baseline_mape_error:.2f}%</b>.
        <br>
        Lower MAPE indicates better model accuracy.
        </div>
        """
    )
    return baseline_mape_error, mape


@app.cell
def _():
    mo.md(
        """
    <div style="text-align: center;">
    <h3>Step 7: ML Model: LightGBM with covariates</h3>
    Now, let's add machine learning! The LightGBM model will utilize past sales and event information (covariates) to improve forecasting accuracy.<br />
    Press the button below to start training the ML model.
    </div>
    """
    )
    return


@app.cell
def _(test, train):
    model = LightGBMModel(
        lags=7,
        lags_past_covariates=7,
        output_chunk_length=7,
        verbose=-1,
        random_state=SEED,
    )

    full_has_event_ts = train["has_event"].concatenate(test["has_event"])

    start_training_ml_model_btn = mo.ui.run_button(label="Start Training ML model")
    mo.hstack([start_training_ml_model_btn], justify="center")
    return full_has_event_ts, model, start_training_ml_model_btn


@app.cell
def _(full_has_event_ts, model, start_training_ml_model_btn, train):
    mo.stop(not start_training_ml_model_btn.value)

    with mo.status.spinner(subtitle="Training ML Model ..."):
        _target = train[  # We can use _ to create private variables only accessible in this cell.
            "sales_qty"
        ]
        model.fit(
            series=_target,
            past_covariates=full_has_event_ts,
        )
    mo.md(
        """
        <div style="text-align: center; color: green; font-weight: bold;">
        ✅ Model training completed successfully!
        </div>
        """
    )
    return


@app.cell
def _(full_has_event_ts, model, start_training_ml_model_btn, test, train):
    mo.stop(not start_training_ml_model_btn.value)

    surpress_warnings()
    ml_model_forecast = model.predict(
        series=train["sales_qty"],
        past_covariates=full_has_event_ts,
        n=31,
    )
    test["sales_qty"].plot(label="Actual")
    ml_model_forecast.plot(title="ML Model Forecast for Next 31 Days")
    return (ml_model_forecast,)


@app.cell
def _(
    baseline_mape_error,
    mape,
    ml_model_forecast,
    start_training_ml_model_btn,
    test,
):
    mo.stop(not start_training_ml_model_btn.value)

    lgbm_mape_error = mape(test["sales_qty"], ml_model_forecast)

    mo.md(
        f"""
        <div style="text-align: center;">
        <h3>ML Model Evaluation</h3>
        The Mean Absolute Percentage Error (MAPE) of the LightGBM model is <b>{lgbm_mape_error:.2f}%</b> (vs <b>{baseline_mape_error:.2f}%</b> of the baseline)!
        </div>
        """
    )
    return


@app.cell
def _(ml_model_forecast, naive_forecast, test):
    test["sales_qty"].plot(label="Actual")
    naive_forecast.plot(label="Baseline Forecast")
    ml_model_forecast.plot(label="ML Model Forecast")
    return


@app.cell
def _(model, start_training_ml_model_btn):
    mo.stop(not start_training_ml_model_btn.value)

    # Save the trained model
    model_filename = "trained_model_2025_07_15.pkl"
    model.save(model_filename)
    mo.md(
        """
        <div style="text-align: center; color: green; font-weight: bold;">
        💾 Model saved successfully!
        </div>
        """
    )
    return


@app.cell
def _():
    mo.md(
        r"""
    <div style="text-align: center;">
      <h3>Step 8: Inference 📈</h3>
      Let's now predict the sales using the trained model. <br><br>
      Note that we need three inputs to this (reusable!) function:
      <br>
      <ul style="display: inline-block; text-align: left;">
        <li>The trained LightGBM model</li>
        <li>The sales data of the past week</li>
        <li>The event data of the past week</li>
      </ul>
      <br>
      Feel free to change these inputs to see how the model reacts!
    </div>
    """
    )
    return


@app.cell
def _(range_slider):
    surpress_warnings()
    import random

    random.seed(SEED)
    past_week_sales = [
        random.randint(range_slider.value[0], range_slider.value[1])
        for _ in range(7)
    ]
    past_week_events = [1, 0, 0, 0, 0, 0, 1]
    return past_week_events, past_week_sales


@app.cell
def _():
    range_slider = mo.ui.range_slider(
        start=0,
        stop=2000,
        step=100,
        value=[200, 400],
        full_width=True,
        show_value=True,
    )
    range_slider
    return (range_slider,)


@app.cell
def _(past_week_events, past_week_sales):
    _predicted_sales = predict_sales(
        load_model(), past_week_sales, past_week_events
    )
    _last_week_sales = past_week_sales[0]
    _direction = "increase" if _predicted_sales > _last_week_sales else "decrease"


    mo.hstack(
        [
            mo.stat(
                value=_predicted_sales,
                label="Tomorrow's sales (predicted)",
                caption=f"{abs(_predicted_sales - _last_week_sales)} from last week",
                direction=_direction,
            )
        ],
        justify="center",
    )
    return


@app.function
def load_model(
    model_filename_input: str = "trained_model_2025_07_15.pkl",
) -> LightGBMModel:
    return LightGBMModel.load(model_filename_input)


@app.function
def predict_sales(
    loaded_model: LightGBMModel,
    _sales_values: list[int],
    _event_values: list[int],
    predicted_date="2025-12-08",
) -> int:
    from darts.utils.utils import generate_index

    _date = pd.to_datetime(predicted_date)
    # Check if date is after November 2025 (end of training data).
    if _date.month < 12 and _date.year < 2025:
        return ValueError("Date is in or before trainig data")

    _input_target_series_min_length = abs(loaded_model.lags["target"][0])
    _input_past_covariate_series_min_length = abs(loaded_model.lags["past"][0])

    _sales_ts = generate_darts_ts(
        values=_sales_values,
        start_date=_date,
        min_input_values_length=_input_target_series_min_length,
    )

    _cov_ts = generate_darts_ts(
        values=_event_values,
        start_date=_date,
        min_input_values_length=_input_target_series_min_length,
    )

    _forecast = loaded_model.predict(
        n=7, series=_sales_ts, past_covariates=_cov_ts
    )
    return int(_forecast[_date].values()[0].item())


@app.function
def generate_darts_ts(
    values: list[int], start_date: str, min_input_values_length: int = 7
):
    from pandas import Series
    from darts.utils.utils import generate_index

    if len(values) < min_input_values_length:
        raise ValueError(
            f"Input should at least be of length {min_input_values_length}"
        )

    _vals, _times = (
        values,
        generate_index(
            start_date - pd.to_timedelta(min_input_values_length, unit="D"),
            length=min_input_values_length,
            freq="D",
        ),
    )
    _pd_series = pd.Series(_vals, index=_times)
    return TimeSeries.from_series(_pd_series)


@app.function
def surpress_warnings():
    import warnings

    warnings.filterwarnings(
        action="ignore",
        message="X does not have valid feature names",
        category=UserWarning,
        module="sklearn.utils.validation",
    )


if __name__ == "__main__":
    app.run()
