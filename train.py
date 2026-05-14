import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data_preparation import (
    load_and_prepare_k1,
    create_features
)

# ====================== ЗАГРУЗКА ======================

df, _ = load_and_prepare_k1('K1_old.xlsx')
df = create_features(df)

# ====================== ПРИЗНАКИ ======================

feature_cols = [
    'GCSD',
    'ADJ.',
    'GCSD_log',
    'ADJ_log',
    'GCSD_x_ADJ',
    'GCSD_sqrt',
    'ADJ_sqrt',
    'PART_LEN',
    'PART_DIGITS',
    'PART_HAS_LETTER'
]

X = df[feature_cols].values
y = df['PLANT STANDARD'].values.reshape(-1, 1)

# ====================== НОРМАЛИЗАЦИЯ ======================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

y_scaler = StandardScaler()
y_scaled = y_scaler.fit_transform(y)

X_train, X_val, y_train, y_val = train_test_split(
    X_scaled,
    y_scaled,
    test_size=0.2,
    random_state=42
)

X_train = torch.FloatTensor(X_train)
X_val = torch.FloatTensor(X_val)

y_train = torch.FloatTensor(y_train)
y_val = torch.FloatTensor(y_val)

# ====================== AI МОДЕЛЬ ======================

class MiniAI(nn.Module):

    def __init__(self, input_size):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),

            nn.Dropout(0.15),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Dropout(0.1),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x)


model = MiniAI(len(feature_cols))

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.002)

# ====================== LOSS HISTORY ======================

train_losses = []
val_losses = []

# ====================== ASCII GRAPH ======================

import numpy as np

def print_loss_graph(train_losses, val_losses, width=60, steps=40):

    print("\n📊 HORIZONTAL LOSS GRAPH\n")

    # сжимаем историю (чтобы не было 2000 строк)
    idxs = np.linspace(0, len(train_losses) - 1, steps).astype(int)

    t_vals = [train_losses[i] for i in idxs]
    v_vals = [val_losses[i] for i in idxs]

    max_loss = max(max(t_vals), max(v_vals))
    min_loss = min(min(t_vals), min(v_vals))

    def norm(x):
        return int((x - min_loss) / (max_loss - min_loss + 1e-8) * width)

    # ось сверху (шкала)
    print("     0" + " " * (width - 10) + f"{max_loss:.4f}")
    print("     |" + "-" * width + "|")

    for i in range(len(t_vals)):

        t = norm(t_vals[i])
        v = norm(v_vals[i])

        line = [" "] * width

        line[t] = "T"
        line[v] = "V"

        # если совпали
        if t == v:
            line[t] = "✪"

        print(f"{i:02d} |{''.join(line)}|")

    print("     |" + "-" * width + "|")
    print("     0" + " " * (width - 10) + f"{min_loss:.4f}")


# ====================== ОБУЧЕНИЕ ======================

print('🚀 Обучение AI...')

for epoch in range(2000):

    model.train()
    optimizer.zero_grad()

    pred = model(X_train)
    loss = criterion(pred, y_train)

    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_pred = model(X_val)
        val_loss = criterion(val_pred, y_val)

    train_losses.append(loss.item())
    val_losses.append(val_loss.item())

    if epoch % 100 == 0:
        print(
            f'Epoch {epoch} | '
            f'Train {loss.item():.6f} | '
            f'Val {val_loss.item():.6f}'
        )

# ====================== ГРАФИК ======================

print_loss_graph(train_losses, val_losses)

# ====================== СОХРАНЕНИЕ ======================

torch.save(
    model.state_dict(),
    'harness_model_k1.pth'
)

with open('feature_cols_k1.pkl', 'wb') as f:
    pickle.dump(feature_cols, f)

with open('scaler_k1.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('y_scaler_k1.pkl', 'wb') as f:
    pickle.dump(y_scaler, f)

print('\n✅ AI модель обучена')