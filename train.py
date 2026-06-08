import os
import datetime
import numpy as np

from data_preparation import (
    load_cutting_sheet,
    load_all_filters,
    compute_detailed_score,
    create_cutting_features,
    _encode_machine_group,
    FEATURE_COLS_CUTTING,
    load_and_prepare_total_gcsd,
    create_features_total_gcsd,
    FEATURE_COLS_TOTAL_GCSD,
    GENERIC_SHEETS,
    SIMPLE_QTY_SHEETS,
    load_sheet_dynamic,
)
from model import MiniAI, log

FILE_PATH   = 'K0_old.xlsx'
MODELS_DIR  = 'models'
MIN_SAMPLES = 5


def safe_name(s):
    return s.replace(' ', '_').replace('/', '_').replace('\n', '_')


def train_and_save(sheet_name, X, y, feature_cols, epochs=3000, sample_weight=None):
    model_dir = os.path.join(MODELS_DIR, safe_name(sheet_name))
    os.makedirs(model_dir, exist_ok=True)

    ai = MiniAI(
        input_size=len(feature_cols),
        hidden=(128, 64, 32),     # глубже — больше мощности
        epochs=epochs,
        lr=0.001,
        log_target=False,         # score уже в линейном масштабе
        batch_size=512,
        patience=500,
        dropout=0.1,
    )
    ai.fit(X, y, feature_cols, verbose=True, sample_weight=sample_weight)
    ai.save(os.path.join(model_dir, 'model.pkl'))
    log(f'  💾 Модель сохранена → {model_dir}/model.pkl')
    return ai


# ═══════════════════════════════════════════════════════════════════════════════

session_start = datetime.datetime.now()
W = 70

log()
log('  ' + '═' * W)
log(f'  {"🚀  TRAINING — Best Wire Predictor (Detailed Score)":^{W}}')
log(f'  {"Started: " + session_start.strftime("%Y-%m-%d  %H:%M:%S"):^{W}}')
log(f'  {"File: " + FILE_PATH:^{W}}')
log('  ' + '═' * W)

all_results = []


# ─── 1. CUTTING — главная модель ──────────────────────────────────────────────

log()
log('  ' + '=' * W)
log('  CUTTING  — обучение предсказывать детальный score')
log('  ' + '=' * W)
log()
log('  Концепт: target = сумма пройденных операций по всем трём пространствам.')
log('  (LEAD PREP + LEAD PREP FA + High Voltage — каждая строка = одна операция)')
log('  Нейросеть предсказывает этот score → лучший провод = max(predicted_score).')
log()

df_cutting = load_cutting_sheet(FILE_PATH, 'CUTTING')
log(f'  Строк в CUTTING: {len(df_cutting)}')

log()
log('  Загрузка фильтров операций...')
filters_dict = load_all_filters(FILE_PATH)

total_ops = sum(len(v) for v in filters_dict.values())
log(f'  Итого операций с ограничениями: {total_ops}')
log()

# ─── Вычисляем target для каждого провода ────────────────────────────────────

log('  Вычисление score для каждого провода...')
scores_total = []
scores_lp    = []
scores_lpfa  = []
scores_hv    = []

for _, w in df_cutting.iterrows():
    mid_g = (w['min_gage'] + w['max_gage']) / 2.0
    mid_l = (w['min_len']  + w['max_len'])  / 2.0
    total, s_lp, s_lpfa, s_hv = compute_detailed_score(mid_g, mid_l, filters_dict)
    scores_total.append(total)
    scores_lp.append(s_lp)
    scores_lpfa.append(s_lpfa)
    scores_hv.append(s_hv)

y_scores = np.array(scores_total, dtype=float)
y_lp     = np.array(scores_lp,    dtype=float)
y_lpfa   = np.array(scores_lpfa,  dtype=float)
y_hv     = np.array(scores_hv,    dtype=float)

log()
log('  Статистика target (детальный score):')
log(f'    Диапазон: {y_scores.min():.0f} – {y_scores.max():.0f}')
log(f'    Среднее:  {y_scores.mean():.1f}  ±  {y_scores.std():.1f}')
log(f'    LP:   min={y_lp.min():.0f} max={y_lp.max():.0f} mean={y_lp.mean():.1f}')
log(f'    LPFA: min={y_lpfa.min():.0f} max={y_lpfa.max():.0f} mean={y_lpfa.mean():.1f}')
log(f'    HV:   min={y_hv.min():.0f} max={y_hv.max():.0f} mean={y_hv.mean():.1f}')
log()

# Распределение
n_bins = 10
bin_edges = np.linspace(y_scores.min(), y_scores.max(), n_bins + 1)
log('  Распределение score:')
for i in range(n_bins):
    lo_b, hi_b = bin_edges[i], bin_edges[i+1]
    cnt = ((y_scores >= lo_b) & (y_scores < hi_b + 0.001)).sum()
    bar = '█' * int(cnt / len(y_scores) * 40)
    log(f'    [{lo_b:5.0f}–{hi_b:5.0f}]: {cnt:4d}  {bar}')
log()

# ─── Фичи + аугментация ──────────────────────────────────────────────────────

_, group_mapping, group_global_mean = _encode_machine_group(df_cutting)
df_feat = create_cutting_features(df_cutting, group_mapping, group_global_mean)
X_orig = df_feat[FEATURE_COLS_CUTTING].values

# Аугментация: добавляем шум для обобщения (у нас 575 строк — маловато)
rng = np.random.default_rng(42)
N_AUG = 15  # сколько раз дублируем с шумом
aug_X_list = [X_orig]
aug_y_list = [y_scores]

for _ in range(N_AUG):
    noise = rng.normal(0, 0.02, X_orig.shape)  # 2% шума
    X_noisy = X_orig + X_orig * noise
    aug_X_list.append(X_noisy)
    aug_y_list.append(y_scores)

X = np.vstack(aug_X_list)
y = np.concatenate(aug_y_list)

log(f'  Обучающая выборка после аугментации: {len(X)} строк')
log(f'  (оригинал: {len(X_orig)}, ×{N_AUG + 1} с Gaussian noise 2%)')
log()

# Взвешиваем: провода с высоким score важнее
y_norm = (y - y.min()) / (y.max() - y.min() + 1e-8)
sample_weight = 1.0 + 4.0 * y_norm  # вес от 1 до 5

log(f'  Веса обучающих примеров: min={sample_weight.min():.2f} max={sample_weight.max():.2f}')
log()

# ─── Обучение ─────────────────────────────────────────────────────────────────

ai_cutting = train_and_save(
    'CUTTING', X, y,
    FEATURE_COLS_CUTTING,
    epochs=5000,
    sample_weight=sample_weight,
)

# Сохраняем вспомогательные данные
ai_cutting.group_mapping      = group_mapping
ai_cutting.group_global_mean  = group_global_mean
ai_cutting.filters_dict       = filters_dict
ai_cutting.df_cutting_meta    = df_cutting[['excel_row', 'machine_group',
                                            'min_gage', 'max_gage',
                                            'min_len',  'max_len',
                                            'gcsp']].reset_index(drop=True)
ai_cutting.save(os.path.join(MODELS_DIR, 'CUTTING', 'model.pkl'))

# Проверка на оригинальных данных
preds_orig = ai_cutting.predict(X_orig)
ranking = preds_orig * 10000.0 - df_cutting['gcsp'].values
best_idx = int(np.argmax(ranking))
best_row = df_cutting.iloc[best_idx]
real_score = int(y_scores[best_idx])

log()
log('  ─── Результат на обучающей выборке ───')
log()
log('  🏆  ЛУЧШИЙ ПРОВОД (предсказание нейросети):')
log(f'    Excel-строка : {int(best_row["excel_row"])}')
log(f'    Группа       : {best_row["machine_group"]}')
log(f'    Gage         : [{best_row["min_gage"]:.2f} – {best_row["max_gage"]:.2f}]')
log(f'    Length       : [{best_row["min_len"]:.0f} – {best_row["max_len"]:.0f}]')
log(f'    GCSP         : {best_row["gcsp"]:.4f}  сек/шт')
log(f'    Предсказанный score: {preds_orig[best_idx]:.2f}')
log(f'    Реальный score:      {real_score}')

# Детализация по пространствам
mid_g = (best_row['min_gage'] + best_row['max_gage']) / 2
mid_l = (best_row['min_len']  + best_row['max_len'])  / 2
_, s_lp, s_lpfa, s_hv = compute_detailed_score(mid_g, mid_l, filters_dict)
log()
log(f'    Детализация реального score:')
log(f'      LEAD PREP:    {s_lp}  из {len(filters_dict["LEAD PREP"])}  операций')
log(f'      LEAD PREP FA: {s_lpfa}  из {len(filters_dict["LEAD PREP FA"])} операций')
log(f'      High Voltage: {s_hv}  из {len(filters_dict["High Voltage"])}  операций')

log()
log('  Топ-10 проводов по предсказанию:')
log(f'  {"Строка":>6} {"Gage мин":>8} {"Gage макс":>9} {"Len мин":>7} {"Len макс":>8}'
    f' {"GCSP":>7} {"NN score":>9} {"Real":>6}')
log('  ' + '─' * 62)
top10 = np.argsort(ranking)[::-1][:10]
for idx in top10:
    r = df_cutting.iloc[idx]
    log(f'  {int(r["excel_row"]):>6} {r["min_gage"]:>8.2f} {r["max_gage"]:>9.2f}'
        f' {r["min_len"]:>7.0f} {r["max_len"]:>8.0f}'
        f' {r["gcsp"]:>7.4f} {preds_orig[idx]:>9.2f} {int(y_scores[idx]):>6}')

mae = float(np.mean(np.abs(preds_orig - y_scores)))
log()
log(f'  MAE (NN vs реальный score): {mae:.3f}  (диапазон score: {y_scores.max() - y_scores.min():.0f})')

m = ai_cutting.history['metrics_all']
all_results.append({
    'sheet': 'CUTTING',
    'n': len(X),
    'mae': m['mae'],
    'r2':  m['r2'],
    'best_wire_row':   int(best_row['excel_row']),
    'best_wire_score': real_score,
    'status': '✅ trained',
})


# ─── 2. TOTAL VALUES GCSD ──────────────────────────────────────────────────────

log()
log('  ' + '=' * W)
log('  Sheet: TOTAL values GCSD')
log('  ' + '=' * W)

df_gcsd, _ = load_and_prepare_total_gcsd(FILE_PATH, 'TOTAL values GCSD')
df_gcsd    = create_features_total_gcsd(df_gcsd)
df_train   = df_gcsd.dropna(subset=['PLANT STANDARD'])

log(f'  Всего строк: {len(df_gcsd)}, для обучения: {len(df_train)}')

if len(df_train) >= MIN_SAMPLES:
    ai_g = MiniAI(input_size=len(FEATURE_COLS_TOTAL_GCSD),
                  hidden=(64, 32, 16), epochs=3000, lr=0.001,
                  log_target=True, batch_size=512, patience=400)
    ai_g.fit(df_train[FEATURE_COLS_TOTAL_GCSD].values,
             df_train['PLANT STANDARD'].values,
             FEATURE_COLS_TOTAL_GCSD, verbose=True)
    model_dir = os.path.join(MODELS_DIR, safe_name('TOTAL values GCSD'))
    os.makedirs(model_dir, exist_ok=True)
    ai_g.save(os.path.join(model_dir, 'model.pkl'))
    log(f'  💾 Сохранено → {model_dir}/model.pkl')
    m = ai_g.history['metrics_all']
    all_results.append({'sheet': 'TOTAL values GCSD', 'n': len(df_train),
                        'mae': m['mae'], 'r2': m['r2'],
                        'best_wire_row': None, 'best_wire_score': None,
                        'status': '✅ trained'})
else:
    log('  ⚠️  Недостаточно данных.')
    all_results.append({'sheet': 'TOTAL values GCSD', 'n': len(df_train),
                        'mae': None, 'r2': None,
                        'best_wire_row': None, 'best_wire_score': None,
                        'status': '⚠️  skipped'})


# ─── 3. GENERIC SHEETS (LEAD PREP, LEAD PREP FA, High Voltage) ────────────────

import pandas as _pd

log()
log('  ' + '=' * W)
log('  Generic Sheets  (LEAD PREP / LEAD PREP FA / High Voltage)')
log('  ' + '=' * W)

for sheet_name in GENERIC_SHEETS:
    log()
    log(f'  Sheet: {sheet_name}')

    df_sheet, _ = load_sheet_dynamic(FILE_PATH, sheet_name)
    real_rows = df_sheet[df_sheet['TOTAL'] > 0].copy()
    slot_rows = df_sheet[df_sheet['TOTAL'] <= 0].copy()
    log(f'  Реальных строк: {len(real_rows)},  слотов (TOTAL=0): {len(slot_rows)}')

    if len(real_rows) < MIN_SAMPLES:
        log(f'  ⚠️  Мало реальных данных, пропуск.')
        all_results.append({'sheet': sheet_name, 'n': 0,
                            'mae': None, 'r2': None,
                            'best_wire_row': None, 'best_wire_score': None,
                            'status': '⚠️  no real data'})
        continue

    med_min_gage = float(df_cutting['min_gage'].median())
    med_max_gage = float(df_cutting['max_gage'].median())
    med_min_len  = float(df_cutting['min_len'].median())
    med_max_len  = float(df_cutting['max_len'].median())

    # Фичи для generic sheets (GCSP + QTY)
    def make_features(df_rows):
        df_f = df_rows.copy()
        df_f['GCSP_log']   = np.log1p(df_f['GCSP'])
        df_f['QTY_log']    = np.log1p(df_f['QTY'].clip(lower=0))
        df_f['GCSP_x_QTY'] = df_f['GCSP'] * df_f['QTY']
        df_f['GCSP_sqrt']  = np.sqrt(df_f['GCSP'].clip(lower=0))
        df_f['QTY_sqrt']   = np.sqrt(df_f['QTY'].clip(lower=0))
        return df_f

    FCOLS = ['GCSP', 'QTY', 'GCSP_log', 'QTY_log', 'GCSP_x_QTY', 'GCSP_sqrt', 'QTY_sqrt']

    # Строим обучающую выборку из реальных строк (×10 аугментация)
    records = []
    rng_g = np.random.default_rng(123)
    for _ in range(10):
        for _, r in real_rows.iterrows():
            noise = rng_g.normal(0, 0.01)
            records.append({'GCSP': r['GCSP'] * (1 + noise), 'QTY': r['QTY'],
                            'TOTAL': r['TOTAL'], 'source': 'real'})
    # Слоты — QTY=1, TOTAL=GCSP (предполагаем)
    for _, r in slot_rows.iterrows():
        records.append({'GCSP': r['GCSP'], 'QTY': 1.0,
                        'TOTAL': r['GCSP'], 'source': 'slot'})

    df_all = _pd.DataFrame(records)
    if len(df_all) < MIN_SAMPLES:
        log(f'  ⚠️  Недостаточно данных после аугментации.')
        all_results.append({'sheet': sheet_name, 'n': 0,
                            'mae': None, 'r2': None,
                            'best_wire_row': None, 'best_wire_score': None,
                            'status': '⚠️  skipped'})
        continue

    df_feat = make_features(df_all)
    X_g = df_feat[FCOLS].values
    y_g = df_feat['TOTAL'].values
    sw_g = np.where(df_feat['source'] == 'real', 20.0, 1.0)

    model_dir = os.path.join(MODELS_DIR, safe_name(sheet_name))
    os.makedirs(model_dir, exist_ok=True)

    ai_gs = MiniAI(input_size=len(FCOLS), hidden=(64, 32, 16),
                   epochs=2000, lr=0.001, log_target=True,
                   batch_size=256, patience=300)
    ai_gs.fit(X_g, y_g, FCOLS, verbose=True, sample_weight=sw_g)
    ai_gs.save(os.path.join(model_dir, 'model.pkl'))
    log(f'  💾 Сохранено → {model_dir}/model.pkl')

    m = ai_gs.history['metrics_all']
    all_results.append({'sheet': sheet_name,
                        'n': len(df_all),
                        'mae': m['mae'], 'r2': m['r2'],
                        'best_wire_row': None, 'best_wire_score': None,
                        'status': '✅ trained'})


# ─── 4. SIMPLE_QTY_SHEETS — без обучения ─────────────────────────────────────

log()
log('  ' + '-' * W)
log(f'  Листы без нейросети (QTY=1 в predict.py): {SIMPLE_QTY_SHEETS}')
log('  ' + '-' * W)


# ─── ИТОГ ─────────────────────────────────────────────────────────────────────

session_elapsed = (datetime.datetime.now() - session_start).total_seconds()
log()
log('  ' + '═' * W)
log(f'  {"📋  SESSION SUMMARY":^{W}}')
log('  ' + '═' * W)
log()

for r in all_results:
    mae_s = f'{r["mae"]:.4f}' if r['mae'] is not None else '  —  '
    r2_s  = f'{r["r2"]:.4f}'  if r['r2']  is not None else '  —  '
    bw    = f'row {r["best_wire_row"]} (score={r["best_wire_score"]})' \
            if r['best_wire_row'] is not None else '—'
    log(f'  {r["sheet"]:<30}  MAE={mae_s}  R²={r2_s}  BestWire={bw}  {r["status"]}')

log()
log(f'  ✅  Обучение завершено за {session_elapsed:.1f} сек')
log('  ' + '═' * W)