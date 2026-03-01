"""
=======================================================
FILE 3: train_model.py
PURPOSE: Build and train the "brain" of the predictor.
         We actually train THREE separate models, one for each goal:
           1. Podium Classifier   → Will this driver finish P1/P2/P3? (Yes/No)
           2. Lap Time Regressor  → What lap time will they set?
           3. Championship Model  → Who will lead the championship?

HOW ML TRAINING WORKS (simple analogy):
  Imagine showing a child 1000 photos of cats and dogs.
  After enough examples, they learn patterns (ears, tails, fur).
  We do the same — we feed the model past race results and it
  learns patterns like "drivers with low Engine_Risk tend to finish higher."
=======================================================
"""

import pandas as pd
import numpy as np
import joblib  # joblib saves/loads trained models to disk (like saving a game)
import os

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error

from data_ingestion import get_season_data
from preprocessing import full_preprocess

# All saved models go in this folder
MODELS_DIR = 'models'
os.makedirs(MODELS_DIR, exist_ok=True)

# These are the columns our models will READ as input
# Race data uses FinishPosition (not BestLapTime) as the core numerical feature
FEATURE_COLS = ['FinishPosition', 'Engine_Risk', 'Team_Tier', 'Aero_Efficiency', 'Driver_Avg_Finish']


# =======================================================
# MODEL 1: PODIUM CLASSIFIER
# Question: "Will this driver finish on the podium?"
# Output: 1 (yes) or 0 (no)
# Algorithm: Random Forest — builds many decision trees and votes.
# =======================================================
def train_podium_classifier(df):
    """
    Trains a classifier to predict whether a driver will finish P1, P2, or P3.

    A 'classifier' outputs a category (podium / not podium), not a number.
    We use RandomForest because it handles small datasets well and is
    resistant to overfitting (memorising training data instead of learning).
    """
    print("\n[MODEL 1] Training Podium Classifier...")

    # X = features (what the model reads)
    # y = target (what the model tries to predict)
    X = df[FEATURE_COLS]
    y = df['IsPodium']  # 1 = podium, 0 = not podium

    # Split: 80% data to train, 20% to test (held back so we can measure accuracy)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(
        n_estimators=200,    # 200 individual decision trees vote together
        max_depth=6,         # Each tree can ask at most 6 questions
        class_weight='balanced',  # Handles imbalance: only 3/20 drivers podium each race
        random_state=42
    )
    model.fit(X_train, y_train)  # THIS is training — the model learns from X_train

    # Evaluate: how accurate is it on data it has NEVER seen?
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"  ✓ Podium Classifier Accuracy: {accuracy:.1%}")

    # Save the model to disk so predict.py can load it later
    path = os.path.join(MODELS_DIR, 'podium_classifier.pkl')
    joblib.dump(model, path)
    print(f"  ✓ Model saved to: {path}")

    return model


# =======================================================
# MODEL 2: LAP TIME REGRESSOR
# Question: "What lap time (in seconds) will this driver set?"
# Output: A number like 91.42
# Algorithm: Random Forest Regressor (same idea, but predicts numbers)
# =======================================================
def train_laptime_regressor(df):
    """
    Trains a regressor to predict a driver's qualifying lap time.

    A 'regressor' outputs a continuous number (e.g., 91.4 seconds).
    This is useful for qualifying predictions and grid position.
    """
    print("\n[MODEL 2] Training Lap Time Regressor...")

    # We only train on rows where we actually have a recorded lap time
    df_clean = df.dropna(subset=["FinishPosition"])

    X = df_clean[FEATURE_COLS]
    y = df_clean["FinishPosition"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=6,
        random_state=42
    )
    model.fit(X_train, y_train)

    # MAE = Mean Absolute Error: on average, how many seconds off are we?
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f"  ✓ Lap Time Regressor MAE: ±{mae:.3f} positions")

    path = os.path.join(MODELS_DIR, 'laptime_regressor.pkl')
    joblib.dump(model, path)
    print(f"  ✓ Model saved to: {path}")

    return model


# =======================================================
# MODEL 3: CHAMPIONSHIP POINTS PREDICTOR
# Question: "How many points will this driver score this race?"
# Output: A number (0, 1, 2, 4, 6, 8, 10, 12, 15, 18, 25)
# This lets us simulate a full season and project standings.
# =======================================================
def train_championship_model(df):
    """
    Trains a regressor to predict points scored per race.
    By running this for every race in the calendar, we can project
    who leads the Drivers' Championship at the end of the year.
    """
    print("\n[MODEL 3] Training Championship Points Model...")

    df_clean = df.dropna(subset=['Points'])
    X = df_clean[FEATURE_COLS]
    y = df_clean['Points']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f"  ✓ Championship Model MAE: ±{mae:.2f} points per race")

    path = os.path.join(MODELS_DIR, 'championship_model.pkl')
    joblib.dump(model, path)
    print(f"  ✓ Model saved to: {path}")

    return model


# =======================================================
# MASTER TRAINING FUNCTION
# Call this once to train and save all 3 models.
# =======================================================
def train_all_models(training_year=2025, rounds=range(1, 10)):
    """
    Full training pipeline:
      1. Download historical race data (2025 season)
      2. Preprocess and feature-engineer
      3. Train all 3 models
      4. Save to disk

    We use 2025 data to train because 2026 is new —
    the model learns patterns from history, then we apply them to 2026.
    """
    print("=" * 55)
    print("  F1 2026 PREDICTOR — MODEL TRAINING")
    print("=" * 55)

    # STEP 1: Download training data
    df = get_season_data(year=training_year, rounds=rounds)

    # STEP 2: Preprocess
    df, scaler = full_preprocess(df)

    # Save the scaler — we MUST use the exact same scaling at prediction time
    scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"\n  ✓ Scaler saved to: {scaler_path}")

    # STEP 3: Train all models
    train_podium_classifier(df)
    train_laptime_regressor(df)
    train_championship_model(df)

    print("\n" + "=" * 55)
    print("  ✅ ALL MODELS TRAINED AND SAVED TO /models/")
    print("  → Now run: python predict.py")
    print("=" * 55)


# Run: python train_model.py
if __name__ == "__main__":
    train_all_models(training_year=2025, rounds=range(1, 10))