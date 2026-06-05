"""
data_pull.py - Phase 1, Step 1
Connect to OpenSky and print the first 20 live aircraft states.
"""

import pandas as pd
from pyopensky.rest import REST

def fmt(value, decimals):
    """Format a number, or return '-' if it's missing (NaN or NA)."""
    if pd.isna(value):
        return "-"
    return f"{value:.{decimals}f}"

def main():
    api = REST()

    print("Requesting live aircraft states from OpenSky...\n")
    try:
        df = api.states()
    except TypeError:
        df = None

    if df is None or df.empty:
        print("No data returned (or rate-limited). Try again in a minute.")
        return

    print(f"Received {len(df)} aircraft. Showing first 20:\n")

    header = f"{'ICAO24':<8} {'CALLSIGN':<10} {'COUNTRY':<16} {'LAT':>9} {'LON':>10} {'ALT(m)':>8} {'SPD(m/s)':>9}"
    print(header)
    print("-" * len(header))

    for _, row in df.head(20).iterrows():
        callsign = (row["callsign"] or "").strip() or "-"
        lat = fmt(row["latitude"], 3)
        lon = fmt(row["longitude"], 3)
        alt = fmt(row["altitude"], 0)
        spd = fmt(row["groundspeed"], 0)

        print(f"{row['icao24']:<8} {callsign:<10} {row['origin_country']:<16} {lat:>9} {lon:>10} {alt:>8} {spd:>9}")

if __name__ == "__main__":
    main()
