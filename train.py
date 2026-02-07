import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from process import StockDataset
from model import StockTransformer


CSV_FILE = "final_training_data.csv"
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.00001
SEQ_LEN = 60

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

train_dataset = StockDataset(CSV_FILE, seq_len = SEQ_LEN, mode = 'train')
val_dataset = StockDataset(CSV_FILE, seq_len = SEQ_LEN, mode = 'val')
train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle = False)
sample_x, _ = train_dataset[0]
num_features = sample_x.shape[1]
model = StockTransformer(num_features = num_features, d_model = 32, nhead = 2, num_layers = 2,
                         dropout = 0.6).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE, weight_decay=1e-5)

best_loss = float('inf')
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        preds = model(x).squeeze(1)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).squeeze(1)
            loss = criterion(preds, y)
            val_loss += loss.item()
    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch + 1}: Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        torch.save(model.state_dict(), "best_model.pth")
        print("  >>> Model Saved!")