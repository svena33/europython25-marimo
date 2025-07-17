# EuroPython 2025 - Meet Marimo

<p align="center">
  <img src="logos/ep2025.png" alt="EuroPython 2025" width="75", height="75", style="margin-right: 25px;"  />
  <img src="logos/marimo_logo.svg" alt="Marimo Logo" width="200", style="margin-left: 25px;"  />
</p>

This repository contains the demonstration materials for the presentation **Meet Marimo: The next-gen notebook** at EuroPython 2025.

## 🎯 About This Presentation

This presentation showcases **Marimo**, an open-source reactive notebook alternative that's reproducible, git-friendly, executable as a script, and shareable as an app.

## 📋 What's Included

### Interactive Demos

- **`EUP25_Marimo_D1.py`** - European Snacks Explorer

  - Interactive dropdown to explore local snacks from different European cities
  - Demonstrates Marimo's UI components and reactivity

- **`EUP25_Marimo_D2.py`** - Trdelník Sales Forecasting

  - Time series forecasting demo using LightGBM
  - Showcases Marimo's integration with machine learning workflows
  - Interactive data visualization with Altair

- **`marimo_inference.py`** - Standalone Model Inference
  - Demonstrates how to use trained models outside of Marimo
  - Shows integration between Marimo notebooks and production code

### Data Assets

The `data/` folder contains:

- **`european_snacks.csv`** - Dataset mapping European cities to their local snacks.
- **`trdelnik_sales.csv`** - Historical sales data (synthetic data) for Trdelník sales forecasting.

### Machine Learning Models

- **`trained_model_2025_07_15.pkl`** - Pre-trained LightGBM model for sales forecasting

## 🚀 Getting Started

### Prerequisites

- Python 3.12
- UV package manager ([installation instructions](https://docs.astral.sh/uv/getting-started/installation/))

### Installation

1. Clone this repository:

```bash
git clone git clone https://github.com/svena33/europython25-marimo
```

2. Install dependencies using UV:

```bash
uv sync
```

### Running the Demos

Launch the interactive demos with Marimo:

```bash
# European Snacks Explorer
uv run marimo edit EUP25_Marimo_D1.py

# Trdelník Sales Forecasting
uv run marimo edit EUP25_Marimo_D2.py
```

Or run the standalone inference example:

```bash
uv run python marimo_inference.py
```

## 🛠 Technologies

- **[Marimo](https://marimo.io/)** - Reactive Python notebooks
- **[Altair](https://altair-viz.github.io/)** - Declarative visualization
- **[Darts](https://unit8co.github.io/darts/)** - Time series forecasting library

## Troubleshooting

The darts might require some extra steps to setup. Follow their installation guide if you run into problems: https://github.com/unit8co/darts/blob/master/INSTALL.md.

## <img src="logos/marimo_logo_small.png" alt="marimo" width="20" /> About Marimo

Marimo is a reactive notebook for Python that automatically updates cells when their dependencies change. It offers:

- **Reactive execution** - No more stale cells or hidden state
- **UI components** - Rich, interactive elements out of the box
- **Git-friendly** - Marimo notebooks are just `.py` files so they work well with version control
- **Production ready** - Deploy notebooks as web apps with a single command

Learn more at [marimo.io](https://marimo.io/)

_Presented at EuroPython 2025 by [Sven Arends](https://svenarends.com)_
