import pandas as pd
import requests
import re
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
from datetime import timedelta


# ==========================
# CONFIG
# ==========================

# CSV with historic crest data (already in your repo)
CSV_PATH = "bogue_chitto_tylertown_historiccrests.csv"

# USGS site and analysis parameters
STATION_ID = "02490500"       # Bogue Chitto River near Tylertown, MS
REFERENCE_STAGE = 15.0        # Stage (ft) where we sample rate-of-rise
MIN_IV_YEAR = 1990            # Skip events before this year (no IV data)


# ==========================
# HELPER FUNCTIONS
# ==========================

def parse_crest_csv(csv_path: str) -> pd.DataFrame:
    """
    Read the historic crests CSV and extract:
      - crest_height (float feet)
      - crest_date (datetime)
    Assumes format like:
       rank, "34.70 ft", "on 02-01-1936", empty
    """

    df = pd.read_csv(
        csv_path,
        header=None,
        names=["rank", "height_str", "date_str", "empty"]
    )

    # Extract numeric height
    def extract_height(text):
        if pd.isna(text):
            return None
        m = re.search(r"[\d\.]+", str(text))
        return float(m.group()) if m else None

    # Extract date
    def extract_date(text):
        if pd.isna(text):
            return None
        m = re.search(r"\d{2}-\d{2}-\d{4}", str(text))
        if not m:
            return None
        return pd.to_datetime(m.group(), format="%m-%d-%Y")

    df["crest_height"] = df["height_str"].apply(extract_height)
    df["crest_date"] = df["date_str"].apply(extract_date)

    df = df.dropna(subset=["crest_height", "crest_date"]).reset_index(drop=True)
    return df


def fetch_usgs_iv(station_id: str, start_dt: str, end_dt: str) -> pd.DataFrame:
    """
    Fetch USGS IV gage height data for a window and return as DataFrame
    indexed by datetime with a 'value' column (feet).
    """
    url = (
        "https://nwis.waterservices.usgs.gov/nwis/iv/"
        f"?site={station_id}"
        f"&startDT={start_dt}"
        f"&endDT={end_dt}"
        "&parameterCd=00065"
        "&format=json"
    )

    response = requests.get(url, timeout=20)
    response.raise_for_status()
    data = response.json()

    ts_list = data.get("value", {}).get("timeSeries", [])
    if not ts_list:
        return pd.DataFrame()

    ts_data = ts_list[0]["values"][0]["value"]
    event_df = pd.DataFrame(ts_data)

    event_df["value"] = pd.to_numeric(event_df["value"], errors="coerce")
    event_df["dateTime"] = pd.to_datetime(event_df["dateTime"])
    event_df = event_df.set_index("dateTime").dropna()

    return event_df


# ==========================
# MAIN ANALYSIS
# ==========================

def main():
    # --- 1. Load historic crest data ---
    df = parse_crest_csv(CSV_PATH)
    print(f"Total historic crest entries in CSV: {len(df)}")

    results = []
    valid_event_count = 0
    skipped_event_count = 0

    print(
        f"\nFetching USGS IV data for events "
        f"({STATION_ID}, reference stage {REFERENCE_STAGE} ft)...\n"
    )

    # --- 2. Loop over each crest event ---
    for _, row in df.iterrows():
        crest_date = row["crest_date"]
        crest_height = row["crest_height"]

        # Skip events before IV record is likely to exist
        if crest_date.year < MIN_IV_YEAR:
            skipped_event_count += 1
            continue

        print(f"Processing crest {crest_date.date()} (height {crest_height:.2f} ft)")

        # 4 days before to 1 day after crest
        start_dt = (crest_date - timedelta(days=4)).strftime("%Y-%m-%d")
        end_dt = (crest_date + timedelta(days=1)).strftime("%Y-%m-%d")

        try:
            event_df = fetch_usgs_iv(STATION_ID, start_dt, end_dt)

            if event_df.empty:
                print("  -> No IV data returned, skipping.")
                skipped_event_count += 1
                continue

            # Resample to hourly and compute hourly rate-of-rise
            hourly_df = event_df.resample("1H").mean()
            hourly_df["rate_of_rise"] = hourly_df["value"].diff()

            # First time river crosses reference stage
            crossed_idx = hourly_df[hourly_df["value"] >= REFERENCE_STAGE].index.min()

            if pd.isna(crossed_idx):
                print(
                    f"  -> Never reached {REFERENCE_STAGE} ft in this window, skipping."
                )
                skipped_event_count += 1
                continue

            rate = hourly_df.loc[crossed_idx, "rate_of_rise"]

            if pd.isna(rate) or rate <= 0:
                print("  -> Rate at crossing is non-positive or NaN, skipping.")
                skipped_event_count += 1
                continue

            # Success for this event
            print(f"  -> Valid event. Rate at {REFERENCE_STAGE} ft = {rate:.3f} ft/hr")
            results.append(
                {
                    "crest_date": crest_date.date(),
                    "crest_height": crest_height,
                    "rate_of_rise_per_hr": rate,
                    "cross_time": crossed_idx,
                }
            )
            valid_event_count += 1

        except Exception as e:
            print(f"  -> ERROR fetching/processing data: {e}")
            skipped_event_count += 1

    print("\n==============================")
    print(f"Valid events used in analysis:   {valid_event_count}")
    print(f"Events skipped (any reason):     {skipped_event_count}")
    print("==============================\n")

    # --- 3. Regression / correlation ---
    results_df = pd.DataFrame(results)

    if results_df.empty:
        print("No valid events with both crest height and rate-of-rise. Nothing to do.")
        return

    x = results_df["rate_of_rise_per_hr"].values
    y = results_df["crest_height"].values

    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    r_squared = r_value**2

    print("--- ANALYSIS RESULTS ---")
    print(f"Events analyzed successfully: {len(results_df)}")
    print(f"Correlation (r):                {r_value:.2f}")
    print(f"Skill / Variance Explained R²: {r_squared:.2f}")
    print(
        f"Regression formula:\n"
        f"  Expected Crest (ft) = {slope:.3f} * Rate_of_Rise(ft/hr) + {intercept:.3f}"
    )
    print(f"P-value for slope: {p_value:.4g}")
    print(f"Std error of slope: {std_err:.3f}\n")

    # --- 4. Scatterplot + trendline ---
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, label="Historic Events")

    line_x = np.linspace(x.min(), x.max(), 100)
    line_y = slope * line_x + intercept
    plt.plot(line_x, line_y, label=f"Best-fit line (R² = {r_squared:.2f})")

    plt.title(
        f"Bogue Chitto near Tylertown:\n"
        f"Crest Height vs Rate of Rise at {REFERENCE_STAGE} ft"
    )
    plt.xlabel("Rate of Rise at reference stage (ft per hour)")
    plt.ylabel("Eventual Crest Height (ft)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
