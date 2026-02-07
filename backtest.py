import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from process import StockDataset
from model import StockTransformer


CSV_FILE = "final_training_data.csv"
MODEL_PATH = "best_model.pth"
SEQ_LEN = 60
INITIAL_CAPITAL = 10000
REBALANCE_DAYS = 20
TOP_N = 3
sector_map = {
    'XLK': 'Technology',
    'XLV': 'Health Care',
    'XLF': 'Financials',
    'XLY': 'Consumer Discretionary',
    'XLP': 'Consumer Staples',
    'XLI': 'Industrials',
    'XLU': 'Utilities',
    'XLE': 'Energy',
    'XLB': 'Materials',
    'SPY': 'S&P 500 Benchmark',
    'BIL_Synthetic': 'T-Bills'
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calc_metrics(series):
    if len(series) == 0: return {}
    total_ret = (series.iloc[-1] / INITIAL_CAPITAL) - 1
    rets = series.pct_change().dropna()
    periods_per_year = 252 / REBALANCE_DAYS
    sharpe = (rets.mean() / rets.std()) * np.sqrt(periods_per_year) if rets.std() > 0 else 0
    cummax = series.cummax()
    dd = (series - cummax) / cummax
    max_dd = dd.min()
    return {
        'total_return': total_ret,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'final': series.iloc[-1]
    }


def run_top_n_backtest():
    print(f"Running Top {TOP_N} Equal Weight Strategy")

    df_raw = pd.read_csv(CSV_FILE)
    df_raw['Date'] = pd.to_datetime(df_raw['Date'])

    spy_df = df_raw[['Date', 'SPY']].drop_duplicates().set_index('Date').sort_index()
    spy_df['Ret'] = spy_df['SPY'].pct_change(periods = 20).shift(-20)

    test_dataset = StockDataset(CSV_FILE, seq_len = SEQ_LEN, mode = 'test')
    test_loader = DataLoader(test_dataset, batch_size = 64, shuffle = False)

    sample_x, _ = test_dataset[0]
    num_features = sample_x.shape[1]

    model = StockTransformer(
        num_features = num_features,
        d_model = 32,
        nhead = 2,
        num_layers = 2,
        dropout = 0.6
    ).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location = device, weights_only = True))
    model.eval()

    predictions = []
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            preds = model(x).squeeze(1).cpu().numpy()
            predictions.extend(preds)

    aligned_data = []
    df_raw.sort_values(['Ticker', 'Date'], inplace = True)
    tickers = df_raw['Ticker'].unique()
    pred_idx = 0

    for ticker in tickers:
        group = df_raw[df_raw['Ticker'] == ticker].copy()
        n = len(group)
        val_end = int(n * 0.85)
        test_slice = group.iloc[val_end:]
        valid_rows = test_slice.iloc[SEQ_LEN:].copy()
        num_preds = len(valid_rows)

        if num_preds > 0:
            current_preds = predictions[pred_idx: pred_idx + num_preds]
            valid_rows['Predicted_Score'] = current_preds
            valid_rows['Forward_20d_Return'] = valid_rows['Close'].pct_change(periods = 20).shift(
                -20)
            aligned_data.append(valid_rows)
            pred_idx += num_preds

    simulation_df = pd.concat(aligned_data)
    simulation_df = simulation_df.dropna(subset = ['Forward_20d_Return'])

    score_matrix = simulation_df.pivot(index = 'Date', columns = 'Ticker',
                                       values = 'Predicted_Score')
    ret_matrix = simulation_df.pivot(index = 'Date', columns = 'Ticker',
                                     values = 'Forward_20d_Return')
    common_dates = score_matrix.index.intersection(ret_matrix.index)
    score_matrix = score_matrix.loc[common_dates]
    ret_matrix = ret_matrix.loc[common_dates]

    all_dates = sorted(common_dates)

    strat_history = []
    strat_dates = []
    curr_val = INITIAL_CAPITAL

    i = 0
    while i < len(all_dates):
        date = all_dates[i]
        scores = score_matrix.loc[date]
        valid_scores = scores.dropna()

        if len(valid_scores) < TOP_N or date not in ret_matrix.index:
            i += 1
            continue

        top_picks = valid_scores.nlargest(TOP_N).index.tolist()
        outpicks = [(i, sector_map[i]) for i in top_picks]
        print(f"Date: {date.strftime('%Y-%m-%d')} | Buys: {outpicks}")
        period_return = ret_matrix.loc[date, top_picks].mean()
        curr_val *= (1 + period_return)
        strat_history.append(curr_val)
        strat_dates.append(date)
        i += REBALANCE_DAYS

    bench_history = []
    bench_dates = []
    curr_bench = INITIAL_CAPITAL
    i = 0
    while i < len(all_dates):
        date = all_dates[i]
        if date in spy_df.index:
            bench_ret = spy_df.loc[date, 'Ret']
            if not np.isnan(bench_ret):
                curr_bench *= (1 + bench_ret)
                bench_history.append(curr_bench)
                bench_dates.append(date)
        i += REBALANCE_DAYS

    strat_series = pd.Series(strat_history, index = strat_dates)
    bench_series = pd.Series(bench_history, index = bench_dates)
    strat_m = calc_metrics(strat_series)
    bench_m = calc_metrics(bench_series)
    start_date = strat_series.index[0].strftime('%Y-%m-%d')
    end_date = strat_series.index[-1].strftime('%Y-%m-%d')
    print(f"Test Period: {start_date} to {end_date}")
    print("our portfolio, benchmark")
    print(f"Total Return: {strat_m['total_return'] * 100}%, {bench_m['total_return'] * 100}%")
    print(f"Sharpe Ratio: {strat_m['sharpe']}, {bench_m['sharpe']}")
    print(f"Max Drawdown: {strat_m['max_dd'] * 100}%, {bench_m['max_dd'] * 100}%")
    print(f"Final Value: ${strat_m['final']}, ${bench_m['final']}")


if __name__ == "__main__":
    run_top_n_backtest()