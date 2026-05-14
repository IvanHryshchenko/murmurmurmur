import pickle
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from data_preparation import (
    load_and_prepare_k1,
    create_features
)

# ====================== МОДЕЛЬ ======================

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


# ====================== ЗАГРУЗКА ======================

with open('feature_cols_k1.pkl', 'rb') as f:
    feature_cols = pickle.load(f)

with open('scaler_k1.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('y_scaler_k1.pkl', 'rb') as f:
    y_scaler = pickle.load(f)

model = MiniAI(len(feature_cols))

model.load_state_dict(
    torch.load(
        'harness_model_k1.pth',
        map_location='cpu'
    )
)

model.eval()

print('✅ AI модель загружена')


# ====================== ДАННЫЕ ======================

df_clean, df_original = load_and_prepare_k1('K1_old.xlsx')

df_features = create_features(df_clean)

if len(df_features) == 0:
    print('❌ Нет данных')
    exit()


# ====================== FEATURES ======================

X = df_features[feature_cols].values
X_scaled = scaler.transform(X)
X_tensor = torch.FloatTensor(X_scaled)


# ====================== PREDICTION ======================

with torch.no_grad():
    pred_scaled = model(X_tensor).numpy()

pred = y_scaler.inverse_transform(pred_scaled).flatten()


# ====================== EXPORT ======================

df_result = df_original.copy()

for i in range(len(pred)):

    row_idx = int(df_clean.iloc[i]['row_index'])

    value = float(pred[i])

    # защита от ошибок данных
    if np.isnan(value) or np.isinf(value):
        value = 0.0

    if value < 0:
        value = 0.0

    df_result.iloc[row_idx, 3] = round(value, 4)


# ====================== SAVE ======================

output_file = 'RESULT_K1.xlsx'

df_result.to_excel(
    output_file,
    index=False,
    header=False
)

print('\n✅ Готово')
print(f'✅ Файл: {output_file}')