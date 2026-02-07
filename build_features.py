import pandas as pd
import numpy as np
import yfinance as yf
import requests
import io


INPUT_FILE = "sector_data.csv"
OUTPUT_FILE = "final_training_data.csv"


df_sectors = pd.read_csv(INPUT_FILE, index_col = 0, parse_dates = True)
df_sectors.index.name = 'Date'
if 'BIL' in df_sectors.columns:
    df_sectors.drop(columns = ['BIL'], inplace = True)
tickers = [c for c in df_sectors.columns if c != 'SPY']


try:
    macro_tickers = ['^TNX', '^VIX', '^IRX']
    macro_data = yf.download(macro_tickers, start = "1999-01-01", progress = False)
    if isinstance(macro_data.columns, pd.MultiIndex):
        macro_data = macro_data.xs('Close', axis = 1, level = 0)
    else:
        macro_data = macro_data[['Close']]
    macro_data.rename(columns = {'^TNX': 'Treasury_10Y', '^VIX': 'VIX'}, inplace = True)
    daily_rate = (macro_data['^IRX'] / 100) / 252
    daily_rate = daily_rate.shift(1).fillna(0)
    synthetic_price = 100 * (1 + daily_rate).cumprod()
    macro_data['BIL_Synthetic'] = synthetic_price
    macro_data.drop(columns = ['^IRX'], inplace = True)
except Exception as e:
    print(f"Yahoo Download Error: {e}")
    exit()


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
        date_col = next(c for c in df.columns if 'date' in c.lower())
        df.set_index(date_col, inplace = True)
        df.index = pd.to_datetime(df.index)
        val_col = [c for c in df.columns if c != date_col][0]
        df.rename(columns = {val_col: name}, inplace = True)
        inflation_frames.append(df)
    except Exception as e:
        print(f"Error fetching {name}: {e}")
if not inflation_frames:
    print("Failed to get Inflation data.")
    exit()

df_inflation = pd.concat(inflation_frames, axis = 1)
df_inflation = df_inflation.interpolate(method = 'time')
df_inflation = df_inflation.shift(1, freq = 'M')
df_inflation['CPI_YoY'] = df_inflation['CPI'].pct_change(12) * 100
df_master = df_sectors.join(macro_data).ffill()
df_master = df_master.join(df_inflation, how = 'left').ffill()
df_master.dropna(inplace = True)
tickers.append('BIL_Synthetic')
print(f"Date range: {df_master.index[0]} to {df_master.index[-1]}")


feature_list = []

def calc_rsi(series, window = 14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    loss = loss.replace(0, np.nan)
    rs = gain / loss
    return 100 - (100 / (1 + rs))



for ticker in tickers:
    cols_needed = [ticker, 'SPY', 'Treasury_10Y', 'VIX', 'CPI_YoY', 'Sticky_CPI']
    available_cols = [c for c in cols_needed if c in df_master.columns]
    if ticker not in available_cols:
        continue
    df_t = df_master[available_cols].copy()
    df_t.rename(columns = {ticker: 'Close'}, inplace = True)
    df_t['Rel_Strength'] = df_t['Close'] / df_t['SPY']
    df_t['Trend_200MA'] = df_t['Close'] / df_t['Close'].rolling(200).mean()
    df_t['Mom_60D'] = df_t['Close'].pct_change(60)
    df_t['Vol_20D'] = df_t['Close'].pct_change().rolling(20).std()
    df_t['RSI'] = calc_rsi(df_t['Close']).fillna(50) / 100.0
    df_t['Real_Rate'] = df_t['Treasury_10Y'] - df_t['CPI_YoY']
    df_t['Sticky_Gap'] = df_t['Sticky_CPI'] - df_t['CPI_YoY']
    df_t['VIX_Norm'] = df_t['VIX'] / 100.0
    df_t['Target_Return_20d'] = df_t['Close'].pct_change(periods = -20)
    df_t['Ticker'] = ticker
    df_t.dropna(inplace = True)
    feature_list.append(df_t)

if feature_list:
    final_df = pd.concat(feature_list)
    final_df.index.name = 'Date'
    final_df.reset_index(inplace = True)
    final_df.set_index(['Date', 'Ticker'], inplace = True)
    final_df.to_csv(OUTPUT_FILE)
    print(f"Saved to {OUTPUT_FILE}")
else:
    print("Error: No data generated.")