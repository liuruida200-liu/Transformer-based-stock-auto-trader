import pandas as pd
import requests
import io


TIINGO_API_KEY = "api_key"

symbols = ['XLK', 'XLV', 'XLF', 'XLY', 'XLP', 'XLI', 'XLU', 'XLE', 'XLB','SPY', 'BIL']


def fetch_data_official():
    print(f"Fetching data for {len(symbols)} symbols")

    all_data = []
    headers = {'Content-Type': 'application/json',
        'Authorization': f'Token {TIINGO_API_KEY}'}

    for ticker in symbols:
        try:
            print(f"----- Downloading {ticker}-----")
            url = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices"
            params = {'startDate': '2000-01-01', 'format': 'csv'}
            response = requests.get(url, headers = headers, params = params)

            if response.status_code == 200:
                df = pd.read_csv(io.StringIO(response.text))
                df = df[['date', 'adjClose']]
                df.rename(columns = {'adjClose': ticker}, inplace = True)
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace = True)
                all_data.append(df)
            else:
                print(f"Error fetching {ticker}: {response.text}")

        except Exception as e:
            print(f"Failed on {ticker}: {e}")

    if not all_data:
        print("No data downloaded. Check API Key.")
        return
    price_data = pd.concat(all_data, axis = 1)
    price_data.to_csv("sector_data.csv")


if __name__ == "__main__":
    fetch_data_official()