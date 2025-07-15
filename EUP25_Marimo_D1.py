import marimo

__generated_with = "0.14.10"
app = marimo.App(width="medium")


@app.cell
def _(fav_snack, mo):
    mo.center(mo.image(src=f"data/{fav_snack}.jpg", width=300, height=300, rounded=True))
    return


@app.cell
def _(city, eu_snacks):
    fav_snack = eu_snacks[eu_snacks["City"]==city].iloc[0]["Local Snack"]
    # fav_snack
    return (fav_snack,)


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    return mo, pd


@app.cell
def _(pd):
    eu_snacks = pd.read_csv("data/european_snacks.csv")
    # eu_snacks
    return (eu_snacks,)


@app.cell
def _(city_dropdown):
    city = city_dropdown.value
    return (city,)


@app.cell
def _(eu_snacks, mo):
    city_dropdown = mo.ui.dropdown(options=list(eu_snacks["City"]), value="Amsterdam")
    # city_dropdown
    return (city_dropdown,)


@app.cell
def _(city_dropdown, fav_snack, mo):
    mo.vstack([fav_snack, city_dropdown], justify="space-between", align="center")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
