# 🔋 Battery Voltage Prediction

A machine-learning project that predicts **Voltage_load** from battery measurements.

## Dataset
The project uses `00005.csv`, containing 429 observations and these six numeric columns:

- Voltage_measured
- Current_measured
- Temperature_measured
- Current_load
- Voltage_load
- Time

## ML workflow

1. Load and inspect the data
2. Check missing values
3. Define features and target
4. Use a chronological 80/20 train-test split
5. Compare:
   - Linear Regression
   - Ridge Regression
   - Lasso Regression
6. Evaluate with MAE, RMSE and R²
7. Save the selected preprocessing + model pipeline with `joblib`
8. Deploy with Streamlit

### Target
`Voltage_load`

### Features
`Voltage_measured`, `Current_measured`, `Temperature_measured`, `Current_load`, `Time`

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

## GitHub structure

```text
battery-voltage-prediction/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── 00005.csv
├── models/
│   ├── battery_voltage_model.joblib
│   └── metadata.json
└── notebooks/
    └── battery_voltage_prediction.ipynb
```

## Important deployment detail

The `.joblib` file contains the complete scikit-learn Pipeline, including `StandardScaler` and the trained regression model. This prevents the deployment code from using a different preprocessing procedure from training.

## Results from the current uploaded CSV

The notebook computes the exact MAE, RMSE and R² on the chronological test set when run.
