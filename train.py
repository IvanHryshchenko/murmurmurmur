"""
train.py — Обучение нейросети по концепту заказчика
-----------------------------------------------------

КОНЦЕПТ (Кирилл Шилохвостов):
  Лист CUTTING, строки I3..I681 — TOTAL Qty Leads.
  
  Алгоритм обучения:
    1. Берём строку i (начиная с I3).
    2. Подставляем QTY=1 → активируем эту строку.
    3. Для этой строки известны: Min Gage, Max Gage, Min Length, Max Length, GCSP.
    4. TOTAL TIME = GCSP * 1 = GCSP.
    5. Обучаем нейросеть предсказывать GCSP по [MinGage, MaxGage, MinLen, MaxLen].
    6. Проверяем: предсказание ≤ табличное GCSP (или совпадает).
    7. Переходим к следующей строке.
  
  Идея: каждая строка таблицы — это одна «точка обучения».
  Нейросеть регрессирует GCSP = f(min_gage, max_gage, min_len, max_len).
  
  После обучения: для любого нового провода с известными параметрами
  нейросеть даёт предсказание времени операции.
  
  Метрика прохождения: pred_gcsp <= table_gcsp для каждой строки.
  Собираем pass/fail по каждой строке и итоговый процент прохождения.
"""

import os
import datetime
import numpy as np
import pandas as _pd

from data_preparation import (
    GENERIC_SHEETS,
    FEATURE_COLS,
    FEATURE_COLS_TOTAL_GCSD,
    FEATURE_COLS_CUTTING,
    FEATURE_COLS_AUGMENTED,
    load_cutting_sheet,
    create_cutting_features,
    load_sheet_dynamic,
    create_features,
    load_and_prepare_total_gcsd,
    create_features_total_gcsd,
    build_augmented_dataset,
    create_features_augmented,
)
from model import MiniAI, log

FILE_PATH   = 'K0_old.xlsx'
MODELS_DIR  = 'models'
MIN_SAMPLES = 5


def safe_name(s):
    return s.replace(' ', '_').replace('/', '_').replace('\n', '_')


def train_and_save(sheet_name, X, y, feature_cols, epochs=2000, sample_weight=None):
    model_dir = os.path.join(MODELS_DIR, safe_name(sheet_name))
    os.makedirs(model_dir, exist_ok=True)

    ai = MiniAI(input_size=len(feature_cols), epochs=epochs, lr=0.001,
                log_target=True, batch_size=2048, patience=300)
    ai.fit(X, y, feature_cols, verbose=True, sample_weight=sample_weight)
    ai.save(os.path.join(model_dir, 'model.pkl'))
    log(f'  💾 Model saved → {model_dir}/model.pkl')
    return ai


def _pass_fail_report(y_true, y_pred, label='Wire-by-wire pass/fail'):
    """
    Проверяем каждую строку: pred <= true (или ≤ true + допуск).
    Собираем метрики: сколько строк прошли проверку.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    # Строгая проверка: pred <= table
    passed_strict = (y_pred <= y_true)
    # Мягкая проверка: pred <= table * 1.05 (5% допуск)
    passed_5pct   = (y_pred <= y_true * 1.05)
    # Совпадение: |pred - true| / true < 2%
    passed_match  = (np.abs(y_pred - y_true) / (np.abs(y_true) + 1e-8) < 0.02)

    n = len(y_true)
    log(f'  ── {label} ──')
    log(f'  Total rows          : {n}')
    log(f'  Passed (pred ≤ ref) : {passed_strict.sum():4d}  ({passed_strict.mean()*100:.1f}%)')
    log(f'  Passed (pred ≤ +5%) : {passed_5pct.sum():4d}  ({passed_5pct.mean()*100:.1f}%)')
    log(f'  Match (±2%)         : {passed_match.sum():4d}  ({passed_match.mean()*100:.1f}%)')
    log(f'  Failed              : {(~passed_strict).sum():4d}  ({(~passed_strict).mean()*100:.1f}%)')

    # Топ-5 худших строк (наибольшее превышение)
    over = y_pred - y_true
    worst_idx = np.argsort(over)[::-1][:5]
    log(f'  Top-5 worst (over-predictions):')
    for idx in worst_idx:
        if over[idx] > 0:
            log(f'    row_local={idx:3d}  ref={y_true[idx]:.4f}  pred={y_pred[idx]:.4f}'
                f'  diff={over[idx]:+.4f}  ({over[idx]/y_true[idx]*100:+.1f}%)')

    return {
        'n': n,
        'pass_strict': int(passed_strict.sum()),
        'pass_strict_pct': float(passed_strict.mean() * 100),
        'pass_5pct': int(passed_5pct.sum()),
        'pass_5pct_pct': float(passed_5pct.mean() * 100),
        'match_pct': float(passed_match.mean() * 100),
    }


# ═══════════════════════════════════════════════════════════════════════════════

session_start = datetime.datetime.now()
W = 70

log()
log('  ' + '═' * W)
log(f'  {"🚀  TRAINING SESSION (Wire-by-Wire Concept)":^{W}}')
log(f'  {"Started: " + session_start.strftime("%Y-%m-%d  %H:%M:%S"):^{W}}')
log(f'  {"File: " + FILE_PATH:^{W}}')
log('  ' + '═' * W)

all_results = []


# ─── 1. CUTTING ────────────────────────────────────────────────────────────────

log()
log('  ' + '=' * W)
log('  Sheet: CUTTING  (Wire-by-Wire Training)')
log('  ' + '=' * W)
log()
log('  Концепт: каждая строка I3..I681 — отдельный провод/операция.')
log('  Подставляем QTY=1, цель = Global Sec/Pc (время на шт).')
log('  Признаки: Min Gage, Max Gage, Min Length, Max Length.')
log()

df_cutting = load_cutting_sheet(FILE_PATH, 'CUTTING')
log(f'  Строк с валидными параметрами (GCSP > 0): {len(df_cutting)}')

if len(df_cutting) >= MIN_SAMPLES:
    df_feat = create_cutting_features(df_cutting)

    X = df_feat[FEATURE_COLS_CUTTING].values
    y = df_cutting['gcsp'].values   # Target: Global Sec/Pc

    log(f'  GCSP — mean={y.mean():.4f}  std={y.std():.4f}'
        f'  min={y.min():.4f}  max={y.max():.4f}')
    log()

    # Показываем первые N строк с QTY=1
    log('  Пример активации строк (QTY=1):')
    for _, r in df_cutting.head(5).iterrows():
        total_time = r['gcsp'] * 1  # QTY=1
        log(f'    Строка {int(r["excel_row"]):3d} | '
            f'Gage [{r["min_gage"]:.1f}–{r["max_gage"]:.1f}] | '
            f'Len [{r["min_len"]:.0f}–{r["max_len"]:.0f}] | '
            f'GCSP={r["gcsp"]:.4f} → TOTAL TIME = {total_time:.4f} мин')
    log()

    ai_cutting = train_and_save(
        'CUTTING',
        X, y,
        FEATURE_COLS_CUTTING,
        epochs=3000,  # больше эпох для wire-by-wire
    )

    # Pass/fail проверка: pred ≤ table_gcsp
    preds_all = ai_cutting.predict(X)
    pf = _pass_fail_report(y, preds_all, 'CUTTING wire-by-wire check')

    m = ai_cutting.history['metrics_all']
    all_results.append({
        'sheet':          'CUTTING',
        'n':              len(df_cutting),
        'mae':            m['mae'],
        'r2':             m['r2'],
        'pass_strict_pct': pf['pass_strict_pct'],
        'pass_5pct_pct':  pf['pass_5pct_pct'],
        'status':         '✅ trained',
    })
else:
    log(f'  ⚠️  Недостаточно данных ({len(df_cutting)} строк, нужно {MIN_SAMPLES}), пропуск.')
    all_results.append({'sheet': 'CUTTING', 'n': len(df_cutting),
                        'mae': None, 'r2': None,
                        'pass_strict_pct': None, 'pass_5pct_pct': None,
                        'status': '⚠️  skipped'})


# ─── 2. TOTAL VALUES GCSD ──────────────────────────────────────────────────────

log()
log('  ' + '=' * W)
log('  Sheet: TOTAL values GCSD')
log('  ' + '=' * W)

df_gcsd, _ = load_and_prepare_total_gcsd(FILE_PATH, 'TOTAL values GCSD')
df_gcsd    = create_features_total_gcsd(df_gcsd)
df_train   = df_gcsd.dropna(subset=['PLANT STANDARD'])

log(f'  Всего строк: {len(df_gcsd)}')
log(f'  Для обучения (с меткой): {len(df_train)}')
log(f'  Без метки (для предсказания): {len(df_gcsd) - len(df_train)}')

if len(df_train) >= MIN_SAMPLES:
    ai = train_and_save(
        'TOTAL values GCSD',
        df_train[FEATURE_COLS_TOTAL_GCSD].values,
        df_train['PLANT STANDARD'].values,
        FEATURE_COLS_TOTAL_GCSD,
    )
    m = ai.history['metrics_all']
    preds = ai.predict(df_train[FEATURE_COLS_TOTAL_GCSD].values)
    pf    = _pass_fail_report(df_train['PLANT STANDARD'].values, preds,
                              'TOTAL values GCSD check')
    all_results.append({
        'sheet': 'TOTAL values GCSD', 'n': len(df_train),
        'mae': m['mae'], 'r2': m['r2'],
        'pass_strict_pct': pf['pass_strict_pct'],
        'pass_5pct_pct': pf['pass_5pct_pct'],
        'status': '✅ trained',
    })
else:
    log(f'  ⚠️  Недостаточно данных ({len(df_train)} строк).')
    all_results.append({'sheet': 'TOTAL values GCSD', 'n': len(df_train),
                        'mae': None, 'r2': None,
                        'pass_strict_pct': None, 'pass_5pct_pct': None,
                        'status': '⚠️  skipped'})


# ─── 3. GENERIC SHEETS (инкрементальное обучение по строкам CUTTING) ─────────
#
# Концепт Кирилла Шилохвостова — точное воспроизведение:
#
#   Для каждого generic-листа (LEAD PREP, FA и т.д.):
#     1. Инициализируем модель — делаем первый fit() на реальных строках
#        (QTY > 0) чтобы зафиксировать нормализацию.
#        Если реальных строк нет — используем все строки с первой строкой CUTTING.
#     2. Итерируемся по строкам CUTTING (строки 3..681):
#        - Берём min_gage, max_gage, min_len, max_len из строки i
#        - Подставляем в слоты листа (QTY=0/None строки), QTY=1
#        - TOTAL = GCSP_слота × 1 = GCSP_слота
#        - Вызываем partial_fit() — дообучаем модель на этих данных
#          (веса НЕ сбрасываются, нормализация зафиксирована)
#     3. После всех 600+ строк — финальная проверка pass/fail

log()
log('  ' + '=' * W)
log('  Generic Sheets  (Incremental CUTTING-by-CUTTING Training)')
log('  ' + '=' * W)
log()
log(f'  Строк CUTTING: {len(df_cutting)}')
log('  Концепт: каждая строка CUTTING активирует слоты листа → partial_fit()')
EPOCHS_PER_CUTTING_ROW = 500
log(f'  Эпох на строку CUTTING: {EPOCHS_PER_CUTTING_ROW}')
log()  # эпох на каждую строку CUTTING

for sheet_name in GENERIC_SHEETS:
    log()
    log('  ' + '=' * W)
    log(f'  Sheet: {sheet_name}  (incremental)')
    log('  ' + '=' * W)

    # Загружаем лист один раз
    df_sheet, _ = load_sheet_dynamic(FILE_PATH, sheet_name)
    real_rows = df_sheet[df_sheet['TOTAL'] > 0].copy()
    slot_rows = df_sheet[df_sheet['TOTAL'] <= 0].copy()

    log(f'  Реальных строк : {len(real_rows)}')
    log(f'  Слотов (QTY=0) : {len(slot_rows)}')
    log(f'  Строк CUTTING  : {len(df_cutting)}')
    log(f'  Итого шагов    : {len(df_cutting)} строк × {len(slot_rows)} слотов × {EPOCHS_PER_CUTTING_ROW} эпох')
    log()

    if len(slot_rows) == 0 and len(real_rows) < MIN_SAMPLES:
        log(f'  ⚠️  Нет данных, пропуск.')
        all_results.append({'sheet': sheet_name, 'n': 0,
                            'mae': None, 'r2': None,
                            'pass_strict_pct': None, 'pass_5pct_pct': None,
                            'status': '⚠️  skipped'})
        continue

    # ── Шаг 1: инициализация модели ──────────────────────────────────────────
    # Строим init-датасет: реальные строки + первая строка CUTTING в слотах
    # (только чтобы зафиксировать нормализацию на правдоподобных данных)

    first_cut = df_cutting.iloc[0]
    init_records = []

    # Реальные строки
    for _, r in real_rows.iterrows():
        init_records.append({
            'GCSP': r['GCSP'], 'QTY': r['QTY'], 'TOTAL': r['TOTAL'],
            'CATEGORY': r['CATEGORY'],
            'min_gage': 0.0, 'max_gage': 0.0, 'min_len': 0.0, 'max_len': 0.0,
            'source': 'real',
        })

    # Слоты с первой строкой CUTTING
    for _, slot in slot_rows.iterrows():
        gcsp_j = float(slot['GCSP'])
        init_records.append({
            'GCSP': gcsp_j, 'QTY': 1.0, 'TOTAL': gcsp_j,
            'CATEGORY': slot['CATEGORY'],
            'min_gage': float(first_cut['min_gage']),
            'max_gage': float(first_cut['max_gage']),
            'min_len':  float(first_cut['min_len']),
            'max_len':  float(first_cut['max_len']),
            'source': 'augmented',
        })

    df_init = _pd.DataFrame(init_records)
    if len(df_init) < MIN_SAMPLES:
        log(f'  ⚠️  Недостаточно данных для инициализации, пропуск.')
        all_results.append({'sheet': sheet_name, 'n': 0,
                            'mae': None, 'r2': None,
                            'pass_strict_pct': None, 'pass_5pct_pct': None,
                            'status': '⚠️  skipped'})
        continue

    df_init_feat = create_features_augmented(df_init)
    X_init = df_init_feat[FEATURE_COLS_AUGMENTED].values
    y_init = df_init_feat['TOTAL'].values
    sw_init = np.where(df_init_feat['source'] == 'real', 20.0, 1.0)

    log('  Инициализация нормализации (первый fit)...')
    model_dir = os.path.join(MODELS_DIR, safe_name(sheet_name))
    os.makedirs(model_dir, exist_ok=True)

    ai = MiniAI(input_size=len(FEATURE_COLS_AUGMENTED),
                epochs=EPOCHS_PER_CUTTING_ROW, lr=0.001,
                log_target=True, batch_size=512, patience=150)
    ai.fit(X_init, y_init, FEATURE_COLS_AUGMENTED, verbose=False,
           sample_weight=sw_init)
    log(f'  Нормализация зафиксирована. Начинаем инкрементальное обучение...')
    log()

    # ── Шаг 2: итерация по строкам CUTTING ────────────────────────────────────
    n_cut = len(df_cutting)
    log_every = max(1, n_cut // 10)  # логируем каждые 10%

    for cut_idx, (_, cut_row) in enumerate(df_cutting.iterrows()):
        c_min_gage = float(cut_row['min_gage'])
        c_max_gage = float(cut_row['max_gage'])
        c_min_len  = float(cut_row['min_len'])
        c_max_len  = float(cut_row['max_len'])

        # Активируем слоты этой строкой CUTTING (QTY=1)
        step_records = []
        for _, slot in slot_rows.iterrows():
            gcsp_j = float(slot['GCSP'])
            step_records.append({
                'GCSP': gcsp_j, 'QTY': 1.0, 'TOTAL': gcsp_j,
                'CATEGORY': slot['CATEGORY'],
                'min_gage': c_min_gage, 'max_gage': c_max_gage,
                'min_len':  c_min_len,  'max_len':  c_max_len,
                'source': 'augmented',
            })

        # Всегда добавляем реальные строки чтобы не забыть их
        for _, r in real_rows.iterrows():
            step_records.append({
                'GCSP': r['GCSP'], 'QTY': r['QTY'], 'TOTAL': r['TOTAL'],
                'CATEGORY': r['CATEGORY'],
                'min_gage': 0.0, 'max_gage': 0.0, 'min_len': 0.0, 'max_len': 0.0,
                'source': 'real',
            })

        df_step = _pd.DataFrame(step_records)
        df_step_feat = create_features_augmented(df_step)
        X_step = df_step_feat[FEATURE_COLS_AUGMENTED].values
        y_step = df_step_feat['TOTAL'].values
        sw_step = np.where(df_step_feat['source'] == 'real', 20.0, 1.0)

        ai.partial_fit(X_step, y_step, epochs=EPOCHS_PER_CUTTING_ROW,
                       sample_weight=sw_step, verbose=False)

        if (cut_idx + 1) % log_every == 0 or cut_idx == n_cut - 1:
            pct = (cut_idx + 1) / n_cut * 100
            log(f'  [{pct:5.1f}%]  CUTTING строка {cut_idx+1}/{n_cut}'
                f'  gage=[{c_min_gage:.1f}–{c_max_gage:.1f}]'
                f'  len=[{c_min_len:.0f}–{c_max_len:.0f}]'
                f'  t={ai.t} шагов Adam')

    log()
    log(f'  Инкрементальное обучение завершено. Всего шагов Adam: {ai.t}')

    # ── Шаг 3: финальная оценка на всех данных ────────────────────────────────
    ai.save(os.path.join(model_dir, 'model.pkl'))
    log(f'  💾 Model saved → {model_dir}/model.pkl')

    # Собираем полный датасет для оценки
    all_records = []
    for _, r in real_rows.iterrows():
        all_records.append({
            'GCSP': r['GCSP'], 'QTY': r['QTY'], 'TOTAL': r['TOTAL'],
            'CATEGORY': r['CATEGORY'],
            'min_gage': 0.0, 'max_gage': 0.0, 'min_len': 0.0, 'max_len': 0.0,
            'source': 'real',
        })
    # Для оценки берём среднюю строку CUTTING как "типичный провод"
    mid_cut = df_cutting.iloc[len(df_cutting)//2]
    for _, slot in slot_rows.iterrows():
        gcsp_j = float(slot['GCSP'])
        all_records.append({
            'GCSP': gcsp_j, 'QTY': 1.0, 'TOTAL': gcsp_j,
            'CATEGORY': slot['CATEGORY'],
            'min_gage': float(mid_cut['min_gage']),
            'max_gage': float(mid_cut['max_gage']),
            'min_len':  float(mid_cut['min_len']),
            'max_len':  float(mid_cut['max_len']),
            'source': 'augmented',
        })

    df_eval = _pd.DataFrame(all_records)
    df_eval_feat = create_features_augmented(df_eval)
    X_eval = df_eval_feat[FEATURE_COLS_AUGMENTED].values
    y_eval = df_eval_feat['TOTAL'].values

    preds = ai.predict(X_eval)
    pf = _pass_fail_report(y_eval, preds, f'{sheet_name} final check')

    # Метрики
    err = preds - y_eval
    mae = float(np.mean(np.abs(err)))
    ss_r = float(np.sum(err**2))
    ss_t = float(np.sum((y_eval - y_eval.mean())**2))
    r2 = 1.0 - ss_r / (ss_t + 1e-10)

    all_results.append({
        'sheet': sheet_name,
        'n': len(df_cutting) * len(slot_rows) + len(real_rows),
        'mae': mae, 'r2': r2,
        'pass_strict_pct': pf['pass_strict_pct'],
        'pass_5pct_pct': pf['pass_5pct_pct'],
        'status': '✅ trained',
    })


# ─── ИТОГОВАЯ ТАБЛИЦА ──────────────────────────────────────────────────────────

session_end     = datetime.datetime.now()
session_elapsed = (session_end - session_start).total_seconds()

log()
log('  ' + '═' * W)
log(f'  {"📋  SESSION SUMMARY":^{W}}')
log('  ' + '═' * W)
log()

col_w = [26, 6, 9, 7, 12, 12, 14]
total_w = sum(col_w) + 16
header = (f'  │  {"Sheet":<{col_w[0]}}'
          f'{"N":>{col_w[1]}}'
          f'{"MAE":>{col_w[2]}}'
          f'{"R²":>{col_w[3]}}'
          f'{"Pass≤ref%":>{col_w[4]}}'
          f'{"Pass≤+5%%":>{col_w[5]}}'
          f'{"Status":>{col_w[6]}}  │')
sep = '  ├' + '─' * total_w + '┤'

log('  ┌' + '─' * total_w + '┐')
log(header)
log(sep)
for r in all_results:
    mae_s  = f'{r["mae"]:.4f}'  if r['mae']  is not None else '  —  '
    r2_s   = f'{r["r2"]:.4f}'   if r['r2']   is not None else '  —  '
    ps_s   = f'{r["pass_strict_pct"]:.1f}%' if r['pass_strict_pct'] is not None else '  —  '
    p5_s   = f'{r["pass_5pct_pct"]:.1f}%'   if r['pass_5pct_pct']   is not None else '  —  '
    log(f'  │  {r["sheet"]:<{col_w[0]}}'
        f'{r["n"]:>{col_w[1]}}'
        f'{mae_s:>{col_w[2]}}'
        f'{r2_s:>{col_w[3]}}'
        f'{ps_s:>{col_w[4]}}'
        f'{p5_s:>{col_w[5]}}'
        f'{r["status"]:>{col_w[6]}}  │')
log('  └' + '─' * total_w + '┘')

trained = [r for r in all_results if r['mae'] is not None]
if trained:
    log()
    log(f'  Средний MAE : {np.mean([r["mae"] for r in trained]):.4f} мин')
    log(f'  Средний R²  : {np.mean([r["r2"] for r in trained]):.4f}')
    log(f'  Среднее Pass≤ref: {np.mean([r["pass_strict_pct"] for r in trained]):.1f}%')
    log(f'  Среднее Pass≤+5%: {np.mean([r["pass_5pct_pct"]   for r in trained]):.1f}%')

log()
log(f'  ✅  Обучение завершено.')
log(f'  ⏱  Всего: {session_elapsed:.1f} сек')
log(f'  📁  Лог: logs/training.log')
log('  ' + '═' * W)
log()