Objective: To outperform the S&P 500 (SPY) by dynamically rotating capital into the strongest market sectors (e.g., Technology, Energy, Healthcare) while moving to Cash (BIL) during market downturns.

Instructions for using the code:
1.	Run get_data.py to download the raw dataset (you will need to register a free account from Tiingo to use their API
2.	Run build_features.py, process.py, train.py(if you want to try your own hyperparameters. if not, skip the training and use the provided best_model.pth) in the exact order. You are free to test out all the hyper parameters in the train.py
3.	Run backtest.py to get the report where the first column is our portfolio and second is benchmark (SPY).
