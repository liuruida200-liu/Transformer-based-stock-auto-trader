import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import joblib


class StockDataset(Dataset):
    def __init__(self, csv_file, seq_len = 60, mode = 'train', scaler = None):
        df = pd.read_csv(csv_file)
        df['Date'] = pd.to_datetime(df['Date'])
        exclude_cols = ['Date', 'Ticker', 'Target_Return_20d']
        self.feature_cols = [c for c in df.columns if c not in exclude_cols]
        target_col = 'Target_Return_20d'
        df.sort_values(['Ticker', 'Date'], inplace = True)
        tickers = df['Ticker'].unique()
        train_data = []
        val_data = []
        test_data = []
        for ticker in tickers:
            group = df[df['Ticker'] == ticker].copy()
            n = len(group)
            train_end = int(n * 0.70)
            val_end = int(n * 0.85)
            train_data.append(group.iloc[:train_end])
            val_data.append(group.iloc[train_end:val_end])
            test_data.append(group.iloc[val_end:])
        df_train = pd.concat(train_data)
        df_val = pd.concat(val_data)
        df_test = pd.concat(test_data)

        if mode == 'train':
            self.scaler = StandardScaler()
            self.scaler.fit(df_train[self.feature_cols])
            joblib.dump(self.scaler, 'scaler.gz')
            data_subset_dict = df_train
        elif mode == 'val':
            if scaler is None:
                self.scaler = joblib.load('scaler.gz')
            else:
                self.scaler = scaler
            data_subset_dict = df_val
        elif mode == 'test':
            if scaler is None:
                self.scaler = joblib.load('scaler.gz')
            else:
                self.scaler = scaler
            data_subset_dict = df_test
        else:
            raise ValueError("Mode must be 'train', 'val', or 'test'")

        data_subset_dict = data_subset_dict.copy()
        data_subset_dict[self.feature_cols] = self.scaler.transform(
            data_subset_dict[self.feature_cols]
        )
        self.samples = []
        for ticker in data_subset_dict['Ticker'].unique():
            group = data_subset_dict[data_subset_dict['Ticker'] == ticker].copy()
            features = group[self.feature_cols].values
            targets = group[target_col].values
            if len(features) < seq_len:
                continue
            for i in range(len(features) - seq_len):
                x = features[i: i + seq_len]
                y = targets[i + seq_len]
                self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype = torch.float32), torch.tensor(y, dtype = torch.float32)


if __name__ == "__main__":
    train_dataset = StockDataset("final_training_data.csv", seq_len = 60, mode = 'train')
    val_dataset = StockDataset("final_training_data.csv", seq_len = 60, mode = 'val')
    test_dataset = StockDataset("final_training_data.csv", seq_len = 60, mode = 'test')
    train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True)
    x_batch, y_batch = next(iter(train_loader))
