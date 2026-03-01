"""
=======================================================
FILE 4: predict.py
PURPOSE: Load the trained models and make predictions for 2026.
         This is the "game day" file — models are already trained,
         now we use them to answer the actual questions:
           • Who will be on the podium?
           • What lap times will drivers set?
           • Who will lead the championship?
=======================================================
"""

import pandas as pd
import numpy as np
import joblib
import os

from data_ingestion import get_testing_data, get_qualifying_data, get_race_results
from preprocessing import apply_2026_features, clean_lap_times

MODELS_DIR = 'models'
FEATURE_COLS = ['FinishPosition', 'Engine_Risk', 'Team_Tier', 'Aero_Efficiency', 'Driver_Avg_Finish']


def load_models():
    """
    Loads all 3 trained models and the scaler from disk.
    If a model file doesn't exist, it tells you to train first.
    """
    print("\n[LOAD] Loading trained models...")
    required = ['podium_classifier.pkl', 'laptime_regressor.pkl', 'championship_model.pkl', 'scaler.pkl']
    for f in required:
        if not os.path.exists(os.path.join(MODELS_DIR, f)):
            raise FileNotFoundError(
                f"  ✗ '{f}' not found. Please run: python train_model.py first!"
            )

    podium_model      = joblib.load(os.path.join(MODELS_DIR, 'podium_classifier.pkl'))
    laptime_model     = joblib.load(os.path.join(MODELS_DIR, 'laptime_regressor.pkl'))
    championship_model = joblib.load(os.path.join(MODELS_DIR, 'championship_model.pkl'))
    scaler            = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))

    print("  ✓ All models loaded successfully.")
    return podium_model, laptime_model, championship_model, scaler


def prepare_input(df, scaler):
    """
    Applies the same feature engineering and scaling used during training.
    IMPORTANT: We must use the SAME scaler from training.
    Using a different scale would be like measuring in inches but the
    model learned in centimetres — the numbers would be meaningless.
    """
    df = apply_2026_features(df)
    if 'BestLapTime' in df.columns:
        df = clean_lap_times(df, 'BestLapTime')
    if 'FinishPosition' in df.columns:
        df['FinishPosition'] = pd.to_numeric(df['FinishPosition'], errors='coerce').fillna(20.0)
    df[FEATURE_COLS] = scaler.transform(df[FEATURE_COLS])
    return df


# =======================================================
# PREDICTION 1: PODIUM PREDICTION
# =======================================================
def predict_podium(df, podium_model, scaler):
    """
    Predicts which drivers will finish P1, P2, or P3.

    Returns a DataFrame ranked by podium probability (highest first).
    'Podium_Probability' = confidence the model has (0.0 = no chance, 1.0 = certain)
    """
    print("\n[PREDICT 1] Podium Prediction...")

    df_input = prepare_input(df.copy(), scaler)
    X = df_input[FEATURE_COLS]

    # predict_proba gives us the probability of each class
    # Column [1] = probability of being on the podium (class = 1)
    df['Podium_Probability'] = podium_model.predict_proba(X)[:, 1]
    df['Podium_Prediction']  = podium_model.predict(X)

    result = df[['Abbreviation', 'TeamName', 'Podium_Probability', 'Podium_Prediction']]
    result = result.sort_values('Podium_Probability', ascending=False).reset_index(drop=True)
    result.index += 1  # Start ranking from 1, not 0

    print(result.to_string())
    return result


# =======================================================
# PREDICTION 2: QUALIFYING LAP TIME PREDICTION
# =======================================================
def predict_lap_times(df, laptime_model, scaler):
    """
    Predicts each driver's expected qualifying lap time.
    Output is sorted fastest → slowest (simulating the grid order).
    """
    print("\n[PREDICT 2] Qualifying Lap Time Prediction...")

    df_input = prepare_input(df.copy(), scaler)
    X = df_input[FEATURE_COLS]

    df['Predicted_LapTime_s'] = laptime_model.predict(X)

    result = df[['Abbreviation', 'TeamName', 'Predicted_LapTime_s']]
    result = result.sort_values('Predicted_LapTime_s', ascending=True).reset_index(drop=True)
    result.index += 1  # P1 = fastest

    # Format seconds nicely into M:SS.mmm format for readability
    result['Predicted_LapTime'] = result['Predicted_LapTime_s'].apply(
        lambda s: f"{int(s // 60)}:{s % 60:06.3f}"
    )

    print(result[['Abbreviation', 'TeamName', 'Predicted_LapTime']].to_string())
    return result


# =======================================================
# PREDICTION 3: CHAMPIONSHIP STANDINGS PROJECTION
# =======================================================
def predict_championship(df, championship_model, scaler, num_races=24):
    """
    Projects the full season championship standings.

    Strategy:
      1. Predict points per race for each driver.
      2. Multiply by number of races in the season.
      3. Sort by total projected points.

    This is a simplified projection — in reality you'd run it race by race.
    """
    print(f"\n[PREDICT 3] Championship Projection ({num_races} races)...")

    df_input = prepare_input(df.copy(), scaler)
    X = df_input[FEATURE_COLS]

    # Points predicted per race
    df['Points_Per_Race'] = championship_model.predict(X).clip(min=0)  # Can't score negative points

    # Project over full season
    df['Projected_Season_Points'] = (df['Points_Per_Race'] * num_races).round(1)

    result = df[['Abbreviation', 'TeamName', 'Points_Per_Race', 'Projected_Season_Points']]
    result = result.sort_values('Projected_Season_Points', ascending=False).reset_index(drop=True)
    result.index += 1

    print(result.to_string())
    return result


# =======================================================
# MASTER PREDICT FUNCTION
# Run all 3 predictions at once using 2026 testing data.
# =======================================================
def run_all_predictions():
    """
    Full prediction pipeline:
      1. Load trained models
      2. Download 2026 pre-season testing data
      3. Run all 3 predictions
      4. Print results
    """
    print("=" * 55)
    print("  F1 2026 PREDICTOR — RACE PREDICTIONS")
    print("=" * 55)

    # Load models
    podium_model, laptime_model, championship_model, scaler = load_models()

    # We predict using the most recent race data available.
    # Try 2026 Round 1 first; fall back to 2025 Round 1 if 2026 isn't available yet.
    print("\n[DATA] Fetching race data for predictions...")
    try:
        df = get_race_results(year=2026, round_number=1)
        if len(df) == 0 or df['Abbreviation'].isna().all():
            raise ValueError("Empty result — session not available yet.")
        print("  ✓ Using 2026 Round 1 data.")
    except Exception as e:
        print(f"  ⚠ 2026 data not available ({e}). Falling back to 2025 Round 1.")
        df = get_race_results(year=2025, round_number=1)

    # Run predictions
    podium_results       = predict_podium(df.copy(), podium_model, scaler)
    laptime_results      = predict_lap_times(df.copy(), laptime_model, scaler)
    championship_results = predict_championship(df.copy(), championship_model, scaler)

    # Save results to CSV files for further analysis
    os.makedirs('predictions', exist_ok=True)
    podium_results.to_csv('predictions/podium_prediction.csv')
    laptime_results.to_csv('predictions/laptime_prediction.csv')
    championship_results.to_csv('predictions/championship_projection.csv')

    print("\n" + "=" * 55)
    print("  ✅ PREDICTIONS COMPLETE")
    print("  → Results saved to /predictions/ folder")
    print("=" * 55)


# Run: python predict.py
if __name__ == "__main__":
    run_all_predictions()