"""
=======================================================
FILE 1: data_ingestion.py
PURPOSE: Connect to the FastF1 API and download raw F1 data.
         Think of this as the "fetch ingredients" step of cooking.
         Everything downstream depends on this file working correctly.
=======================================================
"""

import fastf1
import pandas as pd
import os

# ---------------------------------------------------------
# CACHING SETUP
# FastF1 downloads HUGE files (telemetry, lap data, timing).
# The cache saves them locally so you don't re-download every time.
# First run: slow (downloading). Every run after: fast (reading from disk).
# ---------------------------------------------------------
CACHE_DIR = 'f1_cache'
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)


# =======================================================
# FUNCTION 1: Pre-Season Testing Data
# Used to understand car baseline performance BEFORE Race 1.
# =======================================================
def get_testing_data(year=2026, test_number=1, session_number=3):
    """
    Downloads pre-season testing session data.

    Args:
        year         (int): The F1 season year.
        test_number  (int): Which test event (1 = Bahrain).
        session_number (int): Which day. Day 3 = most representative.

    Returns:
        pd.DataFrame with columns: Abbreviation, TeamName, BestLapTime (seconds)
    """
    print(f"\n[STEP 1] Fetching {year} Pre-Season Testing (Test {test_number}, Day {session_number})...")

    session = fastf1.get_testing_session(year, test_number, session_number)
    session.load()

    # Print available columns so you can see exactly what FastF1 returns
    print(f"  Available columns: {list(session.results.columns)}")

    results = session.results[['Abbreviation', 'TeamName']].copy()

    # FastF1 column name for best lap varies by session/version — try each candidate
    for candidate in ['BestLapTime', 'Q1', 'Time', 'GapToLeader']:
        if candidate in session.results.columns:
            col_data = session.results[candidate]
            # Convert Timedelta to seconds if needed
            if hasattr(col_data, 'dt'):
                results['BestLapTime'] = col_data.dt.total_seconds()
            else:
                results['BestLapTime'] = pd.to_numeric(col_data, errors='coerce')
            print(f"  ✓ Using '{candidate}' as BestLapTime source.")
            break
    else:
        results['BestLapTime'] = float('nan')
        print("  ⚠ No lap time column found — BestLapTime filled with NaN.")

    print(f"  ✓ Loaded {len(results)} drivers from testing.")
    return results


# =======================================================
# FUNCTION 2: Qualifying Session Data
# Saturday session where drivers set their single fastest lap.
# Determines grid position for the race.
# =======================================================
def get_qualifying_data(year=2026, round_number=1):
    """
    Downloads qualifying results for a specific race weekend.

    Args:
        year         (int): Season year.
        round_number (int): Race weekend number (1 = first GP).

    Returns:
        pd.DataFrame with columns: Abbreviation, TeamName, Position, Q1, Q2, Q3 (seconds)
    """
    print(f"\n[STEP 1b] Fetching Qualifying — {year} Round {round_number}...")

    session = fastf1.get_session(year, round_number, 'Q')
    session.load()

    results = session.results[['Abbreviation', 'TeamName', 'Position', 'Q1', 'Q2', 'Q3']].copy()

    for col in ['Q1', 'Q2', 'Q3']:
        results[col] = results[col].dt.total_seconds()

    print(f"  ✓ Loaded Qualifying results for {len(results)} drivers.")
    return results


# =======================================================
# FUNCTION 3: Race Results
# Main event. Gives us finishing positions and points scored.
# =======================================================
def get_race_results(year=2025, round_number=1):
    """
    Downloads full race results for one Grand Prix.

    Args:
        year         (int): Season year.
        round_number (int): Race weekend number.

    Returns:
        pd.DataFrame with: Abbreviation, TeamName, FinishPosition, Points, Status, IsPodium
    """
    print(f"\n[STEP 1c] Fetching Race results — {year} Round {round_number}...")

    session = fastf1.get_session(year, round_number, 'R')
    session.load()

    results = session.results[['Abbreviation', 'TeamName', 'ClassifiedPosition', 'Points', 'Status']].copy()
    results.rename(columns={'ClassifiedPosition': 'FinishPosition'}, inplace=True)

    # FinishPosition can be 'R' (retired), 'DSQ', etc. — convert safely.
    # errors='coerce' turns non-numeric values into NaN instead of crashing.
    results['FinishPosition'] = pd.to_numeric(results['FinishPosition'], errors='coerce')

    # IsPodium: our TARGET variable for the podium classifier.
    # 1 = finished P1/P2/P3, 0 = did not.
    results['IsPodium'] = (results['FinishPosition'] <= 3).astype(int)

    podium_drivers = results[results['IsPodium'] == 1]['Abbreviation'].tolist()
    print(f"  ✓ Race loaded. Podium: {podium_drivers}")
    return results


# =======================================================
# FUNCTION 4: Full Season Dataset
# Loops through multiple races and stacks results together.
# More data = better trained models.
# =======================================================
def get_season_data(year=2025, rounds=range(1, 6)):
    """
    Builds a multi-race dataset from a full season.
    We use 2025 data to TRAIN because 2026 is new and has no race history yet.

    Args:
        year   (int): Season year.
        rounds (iterable): Which round numbers to include.

    Returns:
        pd.DataFrame: All race results stacked into one big table.
    """
    print(f"\n[STEP 1d] Building season dataset — {year}...")
    all_races = []

    for rnd in rounds:
        try:
            df = get_race_results(year=year, round_number=rnd)
            df['Round'] = rnd  # Tag each row with which race it came from
            all_races.append(df)
        except Exception as e:
            # Skip rounds that fail (e.g. data not available) without crashing
            print(f"  ⚠ Skipping Round {rnd}: {e}")

    if not all_races:
        raise ValueError("No race data could be loaded. Check year/rounds parameters.")

    combined = pd.concat(all_races, ignore_index=True)
    print(f"  ✓ Season dataset: {len(combined)} rows across {len(all_races)} races.")
    return combined


# Quick test — run: python data_ingestion.py
if __name__ == "__main__":
    race_df = get_race_results(year=2025, round_number=1)
    print("\n--- RACE RESULTS SAMPLE ---")
    print(race_df.head(10))