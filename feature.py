import pandas as pd
import numpy as np
import yfinance as yf
import requests
import io


# ==========================================
# CONFIGURATION
# ==========================================
INPUT_FILE = "sector_data.csv"  # The file from Tiingo
OUTPUT_FILE = "final_training_data.csv"

# ==========================================
# 1. LOAD SECTOR DATA
# ==========================================
print("STEP 1: Loading Sector Prices...")
try:
    df_sectors = pd.read_csv(INPUT_FILE, index_col = 0, parse_dates = True)
    df_sectors.index = pd.to_datetime(df_sectors.index)

    # Filter for only the tickers we want (exclude SPY/BIL from being targets)
    tickers = [c for c in df_sectors.columns if c not in ['SPY', 'BIL']]
    print(f"Loaded {len(df_sectors)} rows. Tickers: {tickers}")

except FileNotFoundError:
    print(f"CRITICAL: '{INPUT_FILE}' not found. Please run the Tiingo download script first.")
    exit()

# ==========================================
# 2. FETCH MACRO DATA (Yahoo)
# ==========================================
print("STEP 2: Fetching VIX & Treasury Yields...")
try:
    # Fetch VIX and 10Y Yield
    macro_data = yf.download(['^TNX', '^VIX'], start = "2000-01-01", progress = False)

    # Handle MultiIndex (Fix for newer yfinance versions)
    if isinstance(macro_data.columns, pd.MultiIndex):
        macro_data = macro_data.xs('Close', axis = 1, level = 0)
    else:
        macro_data = macro_data[['Close']]

    macro_data.rename(columns = {'^TNX': 'Treasury_10Y', '^VIX': 'VIX'}, inplace = True)

except Exception as e:
    print(f"Yahoo Download Error: {e}")
    exit()

# ==========================================
# 3. FETCH INFLATION DATA (FRED)
# ==========================================
print("STEP 3: Fetching Inflation (CPI)...")

fred_urls = {
    'CPI': 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL',
    'Sticky_CPI': 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=STICKCPIM157SFRBATL'
}

inflation_frames = []
headers = {'User-Agent': 'Mozilla/5.0'}

for name, url in fred_urls.items():
    try:
        r = requests.get(url, headers = headers)
        df = pd.read_csv(io.StringIO(r.text))

        # Smart Column Finding (Handles 'DATE', 'date', 'observation_date')
        date_col = next(c for c in df.columns if 'date' in c.lower())
        df.set_index(date_col, inplace = True)
        df.index = pd.to_datetime(df.index)

        # Rename value column
        val_col = [c for c in df.columns if c != date_col][0]
        df.rename(columns = {val_col: name}, inplace = True)

        inflation_frames.append(df)

    except Exception as e:
        print(f"Error fetching {name}: {e}")

if not inflation_frames:
    print("Failed to get Inflation data.")
    exit()

# Combine CPIs
df_inflation = pd.concat(inflation_frames, axis = 1)

# PATCH THE DATA GAP
df_inflation = df_inflation.interpolate(method = 'time')

# --- FIX: USE 'M' INSTEAD OF 'ME' FOR COMPATIBILITY ---
# Shift forward 1 month so 'Jan' data is only available in 'Feb'
df_inflation = df_inflation.shift(1, freq = 'M')

# Calculate YoY Inflation
df_inflation['CPI_YoY'] = df_inflation['CPI'].pct_change(12) * 100

# ==========================================
# 4. MERGE DATASETS
# ==========================================
print("STEP 4: Merging all data...")

# Merge Daily Data
df_master = df_sectors.join(macro_data).ffill()

# Merge Monthly Data (Propagate last known value forward)
df_master = df_master.join(df_inflation, how = 'left').ffill()

# Drop startup period
df_master.dropna(inplace = True)


# ==========================================
# 5. FEATURE & TARGET ENGINEERING (FIXED)
# ==========================================
print("STEP 5: Engineering Features & Targets...")

feature_list = []


def calc_rsi(series, window = 14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


for ticker in tickers:
    # Copy relevant columns (checking existence first)
    cols = [ticker, 'SPY', 'Treasury_10Y', 'VIX', 'CPI_YoY', 'Sticky_CPI']
    cols = [c for c in cols if c in df_master.columns]

    df_t = df_master[cols].copy()
    df_t.rename(columns = {ticker: 'Close'}, inplace = True)

    # --- FEATURES (X) ---
    df_t['Rel_Strength'] = df_t['Close'] / df_t['SPY']
    df_t['Trend_200MA'] = df_t['Close'] / df_t['Close'].rolling(200).mean()
    df_t['Mom_60D'] = df_t['Close'].pct_change(60)
    df_t['Vol_20D'] = df_t['Close'].pct_change().rolling(20).std()
    df_t['RSI'] = calc_rsi(df_t['Close']) / 100.0

    # Macros
    if 'Treasury_10Y' in df_t.columns and 'CPI_YoY' in df_t.columns:
        df_t['Real_Rate'] = df_t['Treasury_10Y'] - df_t['CPI_YoY']

    if 'Sticky_CPI' in df_t.columns and 'CPI_YoY' in df_t.columns:
        df_t['Sticky_Gap'] = df_t['Sticky_CPI'] - df_t['CPI_YoY']

    if 'VIX' in df_t.columns:
        df_t['VIX_Norm'] = df_t['VIX'] / 100.0

    # --- TARGET (Y) ---
    # Prediction: Return over NEXT 20 days
    df_t['Target_Return_20d'] = df_t['Close'].pct_change(20).shift(-20)

    df_t['Ticker'] = ticker
    df_t.dropna(inplace = True)

    feature_list.append(df_t)

# Combine and Save
if feature_list:
    final_df = pd.concat(feature_list)

    # --- FIX: FORCE INDEX NAME ---
    # We force the index to be named 'Date' before resetting
    final_df.index.name = 'Date'

    # Now reset_index will definitely create a column named 'Date'
    final_df.reset_index(inplace = True)

    # Now this works because we are 100% sure 'Date' exists
    final_df.set_index(['Date', 'Ticker'], inplace = True)

    print("-" * 30)
    print(f"SUCCESS! Dataset Shape: {final_df.shape}")
    print(final_df[['Close', 'Target_Return_20d', 'Real_Rate']].tail())
    print("-" * 30)

    final_df.to_csv(OUTPUT_FILE)
    print(f"Saved to {OUTPUT_FILE}")
else:
    print("Error: No data generated.")