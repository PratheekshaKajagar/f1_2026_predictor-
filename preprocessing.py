"""
=======================================================
FILE 2: preprocessing.py
PURPOSE: Clean and enrich raw data with F1 domain knowledge.
         Think of this as "prepping ingredients before cooking."
         ML models can't learn from messy or incomplete data.
=======================================================
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# =======================================================
# 2026-SPECIFIC KNOWLEDGE MAPS
# These are hand-crafted scores based on current F1 news.
# They translate qualitative news ("Honda has battery issues")
# into numbers the model can understand.
# =======================================================

# Engine Risk: How reliable is each team's Power Unit?
# Scale: 1 (very safe) → 5 (high risk / brand new)
ENGINE_RISK_MAP = {
    'Mercedes':     1,  # Established works team, most testing mileage
    'Ferrari':      1,  # Strong baseline, reliable
    'McLaren':      1,  # Mercedes customer - benefits from works reliability
    'Williams':     2,  # Mercedes customer - less resources to exploit it
    'Red Bull':     3,  # New Ford partnership (Ford returning after 20+ years)
    'AlphaTauri':   3,  # Same Ford PU as Red Bull
    'Alpine':       2,  # Renault works - mature but not dominant
    'Haas':         2,  # Ferrari customer - reliable PU
    'Aston Martin': 5,  # Honda partnership - suffered battery vibration issues in testing
    'Audi':         4,  # Formerly Sauber - brand new factory team and PU
    'Cadillac':     5,  # Brand new 11th team, Ferrari customer PUs, zero race history
}

# Constructor Tier: Overall car competitiveness estimate entering 2026
# Scale: 1 (top tier) → 4 (backmarker)
TEAM_TIER_MAP = {
    'Mercedes':     1,
    'Ferrari':      1,
    'McLaren':      1,
    'Red Bull':     2,
    'Aston Martin': 2,
    'Alpine':       3,
    'Williams':     3,
    'AlphaTauri':   3,
    'Haas':         3,
    'Audi':         4,
    'Cadillac':     4,
}

# Active Aero Efficiency: 2026 introduces "X-Mode" (low drag) and "Z-Mode" (high downforce).
# Teams that mastered both in testing get an efficiency bonus.
AERO_EFFICIENCY_MAP = {
    'Mercedes':     1.25,
    'Ferrari':      1.20,
    'McLaren':      1.15,
    'Red Bull':     1.10,
    'Aston Martin': 1.00,
    'Alpine':       1.00,
    'Williams':     0.95,
    'AlphaTauri':   0.95,
    'Haas':         0.90,
    'Audi':         0.85,
    'Cadillac':     0.80,
}

# Driver history: approximate average finishing position from 2025 season
# Lower number = better (P1 is best)
DRIVER_AVG_FINISH = {
    'VER': 2.1, 'NOR': 4.2, 'LEC': 5.0, 'HAM': 5.5,
    'PIA': 5.8, 'RUS': 6.0, 'SAI': 6.5, 'ALO': 7.0,
    'GAS': 9.0, 'STR': 11.0,
}


def apply_2026_features(df):
    """
    Adds new computed columns (features) using 2026 domain knowledge.
    Each new column teaches the model something it can't learn from lap times alone.
    """
    print("\n[STEP 2] Applying 2026 Feature Engineering...")

    df['Engine_Risk']     = df['TeamName'].map(ENGINE_RISK_MAP).fillna(3)
    df['Team_Tier']       = df['TeamName'].map(TEAM_TIER_MAP).fillna(3)
    df['Aero_Efficiency'] = df['TeamName'].map(AERO_EFFICIENCY_MAP).fillna(1.0)
    df['Driver_Avg_Finish'] = df['Abbreviation'].map(DRIVER_AVG_FINISH).fillna(12.0)

    print("  ✓ Added: Engine_Risk, Team_Tier, Aero_Efficiency, Driver_Avg_Finish")
    return df


def clean_lap_times(df, time_column='BestLapTime'):
    """
    Handles missing lap times.
    A driver who crashed has no time recorded — we can't just delete them.
    We assign a 'penalty time' so the model knows they underperformed.
    """
    print(f"\n[STEP 2b] Cleaning '{time_column}' column...")
    null_count = df[time_column].isna().sum()
    if null_count > 0:
        penalty = df[time_column].max() + 10.0
        df[time_column] = df[time_column].fillna(penalty)
        print(f"  ✓ Filled {null_count} missing times with penalty: {penalty:.2f}s")
    else:
        print("  ✓ No missing times found.")
    return df


def scale_features(df, feature_cols):
    """
    Scales all numerical features to a similar range (e.g., 0 to 1).

    WHY THIS MATTERS:
    'BestLapTime' might be ~90 seconds. 'Engine_Risk' is 1-5.
    Without scaling, the model thinks lap time is 18x more important
    just because the number is bigger. Scaling fixes this.

    Returns:
        df (pd.DataFrame): With scaled feature columns.
        scaler (StandardScaler): Save this to use the same scale at prediction time.
    """
    print(f"\n[STEP 2c] Scaling features: {feature_cols}")
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    print("  ✓ Features scaled to mean=0, std=1")
    return df, scaler


def full_preprocess(df):
    """
    Master function: runs all preprocessing steps in order.
    Call this instead of calling each function individually.

    Automatically detects whether this is race data or testing/qualifying data
    by checking which columns are present.

    Args:
        df (pd.DataFrame): Raw data from data_ingestion.py

    Returns:
        df (pd.DataFrame): Fully cleaned and feature-engineered data.
        scaler: The scaler object (needed later for predictions).
    """
    df = apply_2026_features(df)

    # Race results don't have BestLapTime — only testing/qualifying data does.
    # We only clean lap times if that column actually exists.
    if 'BestLapTime' in df.columns:
        df = clean_lap_times(df, 'BestLapTime')
        FEATURE_COLS = ['BestLapTime', 'Engine_Risk', 'Team_Tier', 'Aero_Efficiency', 'Driver_Avg_Finish']
    else:
        # Race data: use FinishPosition as the numerical feature instead
        df['FinishPosition'] = pd.to_numeric(df['FinishPosition'], errors='coerce').fillna(20.0)
        FEATURE_COLS = ['FinishPosition', 'Engine_Risk', 'Team_Tier', 'Aero_Efficiency', 'Driver_Avg_Finish']

    df, scaler = scale_features(df, FEATURE_COLS)

    print("\n  ✅ Preprocessing complete.")
    return df, scaler


# Quick test — run: python preprocessing.py
if __name__ == "__main__":
    sample = pd.DataFrame({
        'Abbreviation': ['VER', 'HAM', 'NEW'],
        'TeamName':     ['Red Bull', 'Mercedes', 'Cadillac'],
        'BestLapTime':  [91.5, 91.8, None]
    })
    result, scaler = full_preprocess(sample)
    print("\n--- PREPROCESSED SAMPLE ---")
    print(result)