"""
predict.py — Предсказание лучшего провода нейросетью
-----------------------------------------------------

КОНЦЕПЦИЯ:
  Нейросеть обучена предсказывать детальный score (сумма пройденных операций
  из LEAD PREP + LEAD PREP FA + High Voltage) для каждого провода из CUTTING.
  
  Алгоритм:
  1. Загружаем все провода из CUTTING (575 строк)
  2. Нейросеть предсказывает score для каждого
  3. Выбираем провод с максимальным predicted_score
  4. При равенстве score — выбираем провод с наименьшим GCSP (быстрее)
  5. Записываем результат в Excel: QTY=1 для лучшего, predicted_score в столбец J,
     отметка BEST в столбце K
"""

import os
import datetime
import numpy as np
from openpyxl import load_workbook

from data_preparation import (
    load_cutting_sheet,
    load_all_filters,
    compute_detailed_score,
    create_cutting_features,
    _encode_machine_group,
    FEATURE_COLS_CUTTING,
    GENERIC_SHEETS,
    SIMPLE_QTY_SHEETS,
    load_sheet_dynamic,
    load_and_prepare_total_gcsd,
    create_features_total_gcsd,
    FEATURE_COLS_TOTAL_GCSD,
    _is_header, _to_float, _col_index,
)
from model import MiniAI

FILE_PATH  = 'K0_old.xlsx'
MODELS_DIR = 'models'
OUTPUT_DIR = 'results'
NUMBER_FMT = '0.0000'

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs('logs', exist_ok=True)

_log_file = open(os.path.join('logs', 'predict.log'), 'a', encoding='utf-8')


def log(text=''):
    print(text)
    _log_file.write(text + '\n')
    _log_file.flush()


def safe_name(s):
    return s.replace(' ', '_').replace('/', '_').replace('\n', '_')


def model_path(sheet_name):
    return os.path.join(MODELS_DIR, safe_name(sheet_name), 'model.pkl')


def _round4(v):
    return round(float(v), 4)


def _write_cell(cell, value, fmt=NUMBER_FMT):
    cell.value = _round4(value)
    cell.number_format = fmt


# ═══════════════════════════════════════════════════════════════════════════════

session_start = datetime.datetime.now()
W = 70

log()
log('  ' + '═' * W)
log(f'  {"🔮  BEST WIRE PREDICTION (Neural Network — Detailed Score)":^{W}}')
log(f'  {"Started: " + session_start.strftime("%Y-%m-%d  %H:%M:%S"):^{W}}')
log('  ' + '═' * W)

all_results = []
wb = load_workbook(FILE_PATH)


# ─── 1. CUTTING — поиск лучшего провода ──────────────────────────────────────

log()
log('  ' + '=' * W)
log('  CUTTING — нейросеть предсказывает детальный score для каждого провода')
log('  ' + '=' * W)
log()

df_cutting = load_cutting_sheet(FILE_PATH, 'CUTTING')
log(f'  Загружено {len(df_cutting)} проводов из CUTTING')

# Загружаем фильтры (для верификации реального score)
log()
log('  Загрузка фильтров операций...')
filters_dict = load_all_filters(FILE_PATH)
total_ops = sum(len(v) for v in filters_dict.values())
log(f'  Итого операций с ограничениями: {total_ops}')

# Вычисляем реальный score для каждого провода (верификация)
real_scores = []
real_lp = []
real_lpfa = []
real_hv = []
for _, w in df_cutting.iterrows():
    mid_g = (w['min_gage'] + w['max_gage']) / 2.0
    mid_l = (w['min_len']  + w['max_len'])  / 2.0
    total, s_lp, s_lpfa, s_hv = compute_detailed_score(mid_g, mid_l, filters_dict)
    real_scores.append(total)
    real_lp.append(s_lp)
    real_lpfa.append(s_lpfa)
    real_hv.append(s_hv)

real_scores = np.array(real_scores, dtype=float)

log()
log(f'  Реальный score: min={real_scores.min():.0f}  max={real_scores.max():.0f}'
    f'  mean={real_scores.mean():.1f}')

# ─── Нейросеть ───────────────────────────────────────────────────────────────

mp     = model_path('CUTTING')
use_ai = os.path.exists(mp)

if use_ai:
    ai_cut = MiniAI.load(mp)

    if hasattr(ai_cut, 'group_mapping') and ai_cut.group_mapping:
        gmap, ggmean = ai_cut.group_mapping, ai_cut.group_global_mean or 0.0
    else:
        _, gmap, ggmean = _encode_machine_group(df_cutting)

    df_feat = create_cutting_features(df_cutting, gmap, ggmean)
    X       = df_feat[FEATURE_COLS_CUTTING].values

    nn_scores = ai_cut.predict(X)

    mae = float(np.mean(np.abs(nn_scores - real_scores)))
    log()
    log(f'  ✅ Нейросеть предсказала score для {len(nn_scores)} проводов')
    log(f'     mean={nn_scores.mean():.2f}  min={nn_scores.min():.2f}  max={nn_scores.max():.2f}')
    log(f'     MAE vs реальный score: {mae:.3f}')
else:
    # Fallback: используем реальный score напрямую
    log('  ⚠️  Модель не найдена — fallback (реальный score из правил)')
    nn_scores = real_scores.copy()

# ─── ВЫБОР ЛУЧШЕГО ПРОВОДА ────────────────────────────────────────────────────
# Критерий: max(predicted_score), при равенстве — min(gcsp)

gcsp_arr = df_cutting['gcsp'].values
ranking  = nn_scores * 10000.0 - gcsp_arr   # × 10000 гарантирует приоритет score над gcsp
best_idx = int(np.argmax(ranking))
best_row = df_cutting.iloc[best_idx]

# Детализация реального score для лучшего провода
best_mid_g = (best_row['min_gage'] + best_row['max_gage']) / 2
best_mid_l = (best_row['min_len']  + best_row['max_len'])  / 2
_, best_lp, best_lpfa, best_hv = compute_detailed_score(
    best_mid_g, best_mid_l, filters_dict)

log()
log('  ' + '─' * W)
log(f'  {"🏆  ЛУЧШИЙ ПРОВОД":^{W}}')
log('  ' + '─' * W)
log(f'  Excel-строка      : {int(best_row["excel_row"])}')
log(f'  Группа машины     : {best_row["machine_group"]}')
log(f'  Gage              : [{best_row["min_gage"]:.2f} – {best_row["max_gage"]:.2f}]  (mid={best_mid_g:.2f})')
log(f'  Length            : [{best_row["min_len"]:.0f} – {best_row["max_len"]:.0f}]  (mid={best_mid_l:.0f})')
log(f'  GCSP              : {best_row["gcsp"]:.4f}  сек/шт')
log()
log(f'  NN predicted score: {nn_scores[best_idx]:.2f}')
log(f'  Реальный score    : {int(real_scores[best_idx])}  из {total_ops} операций')
log()
log(f'  Детализация реального score:')
log(f'    LEAD PREP:    {best_lp:4d}  из {len(filters_dict["LEAD PREP"])}  операций')
log(f'    LEAD PREP FA: {best_lpfa:4d}  из {len(filters_dict["LEAD PREP FA"])} операций')
log(f'    High Voltage: {best_hv:4d}  из {len(filters_dict["High Voltage"])}  операций')

# ─── Топ-10 ───────────────────────────────────────────────────────────────────

log()
log('  Топ-10 проводов по предсказанию нейросети:')
log(f'  {"Строка":>6} {"Gage мин":>8} {"Gage макс":>9} {"Len мин":>7} {"Len макс":>8}'
    f' {"GCSP":>7} {"NN score":>9} {"Real":>6}')
log('  ' + '─' * 62)
top10_idx = np.argsort(ranking)[::-1][:10]
for idx in top10_idx:
    r = df_cutting.iloc[idx]
    log(f'  {int(r["excel_row"]):>6} {r["min_gage"]:>8.2f} {r["max_gage"]:>9.2f}'
        f' {r["min_len"]:>7.0f} {r["max_len"]:>8.0f}'
        f' {r["gcsp"]:>7.4f} {nn_scores[idx]:>9.2f} {int(real_scores[idx]):>6}')

# ─── Записываем в Excel ───────────────────────────────────────────────────────

ws_cut = wb['CUTTING']
written_cut = 0

for idx, (_, row_data) in enumerate(df_cutting.iterrows()):
    excel_row  = int(row_data['excel_row'])
    pred_score = float(nn_scores[idx])

    # Столбец J = predicted score (аудит)
    _write_cell(ws_cut.cell(row=excel_row, column=10), pred_score)
    # Столбец L = реальный score (верификация)
    ws_cut.cell(row=excel_row, column=12).value = int(real_scores[idx])
    written_cut += 1

# Лучший провод: QTY=1 в столбец I, BEST в столбец K
best_excel_row = int(best_row['excel_row'])
ws_cut.cell(row=best_excel_row, column=9).value  = 1
ws_cut.cell(row=best_excel_row, column=9).number_format = '0'
ws_cut.cell(row=best_excel_row, column=11).value = 'BEST'

log()
log(f'  ✅ Записано {written_cut} строк.')
log(f'     Лучший провод отмечен QTY=1 (col I) и BEST (col K) → строка {best_excel_row}')

all_results.append(dict(
    sheet='CUTTING', rows=len(df_cutting), written=written_cut,
    method='AI' if use_ai else 'fallback',
    best_wire_row=best_excel_row,
    best_nn_score=float(nn_scores[best_idx]),
    best_real_score=int(real_scores[best_idx]),
))


# ─── 2. TOTAL VALUES GCSD ──────────────────────────────────────────────────────

log()
log('  ' + '=' * W)
log('  Sheet: TOTAL values GCSD')
log('  ' + '=' * W)

df_gcsd, _ = load_and_prepare_total_gcsd(FILE_PATH, 'TOTAL values GCSD')
df_feat_g  = create_features_total_gcsd(df_gcsd)

mp     = model_path('TOTAL values GCSD')
use_ai = os.path.exists(mp)

if use_ai:
    ai    = MiniAI.load(mp)
    X     = df_feat_g[ai.feature_cols].values
    preds = ai.predict(X)
    log(f'  ✅ AI — {len(preds)} предсказаний')
else:
    preds = (df_gcsd['GCSD'] * df_gcsd['ADJ.']).clip(lower=0).values
    log('  ⚠️  Нет модели — fallback GCSD×ADJ')

ws_g   = wb['TOTAL values GCSD']
E_COL, M_COL = 5, 13

written_gcsd = 0
for i, pred in enumerate(preds):
    row_num = int(df_gcsd.iloc[i]['row_index']) + 1
    col_1   = int(df_gcsd.iloc[i]['plant_col']) + 1
    _write_cell(ws_g.cell(row=row_num, column=col_1), pred)
    written_gcsd += 1

ws_g.cell(row=13, column=E_COL).value = '=SUM(E6:E12)'
ws_g.cell(row=23, column=E_COL).value = '=SUM(E19:E22)'
ws_g.cell(row=27, column=M_COL).value = '=SUM(M19:M26)'

log(f'  ✅ Записано {written_gcsd} ячеек')
all_results.append(dict(sheet='TOTAL values GCSD', rows=len(df_gcsd),
                        written=written_gcsd, method='AI' if use_ai else 'fallback',
                        best_wire_row=None, best_nn_score=None, best_real_score=None))


# ─── 3. GENERIC SHEETS (LEAD PREP / LEAD PREP FA / High Voltage) ──────────────

for sheet_name in GENERIC_SHEETS:
    log()
    log(f'  Sheet: {sheet_name}')

    df, _ = load_sheet_dynamic(FILE_PATH, sheet_name)
    if len(df) == 0:
        log('  ⚠️  Нет данных.')
        all_results.append(dict(sheet=sheet_name, rows=0, written=0,
                                method='—', best_wire_row=None,
                                best_nn_score=None, best_real_score=None))
        continue

    FCOLS = ['GCSP', 'QTY', 'GCSP_log', 'QTY_log', 'GCSP_x_QTY', 'GCSP_sqrt', 'QTY_sqrt']

    def _make_feat(df_rows):
        d = df_rows.copy()
        d['GCSP_log']   = np.log1p(d['GCSP'])
        d['QTY_log']    = np.log1p(d['QTY'].clip(lower=0))
        d['GCSP_x_QTY'] = d['GCSP'] * d['QTY']
        d['GCSP_sqrt']  = np.sqrt(d['GCSP'].clip(lower=0))
        d['QTY_sqrt']   = np.sqrt(d['QTY'].clip(lower=0))
        return d

    df_feat = _make_feat(df)
    mp      = model_path(sheet_name)
    use_ai  = os.path.exists(mp)

    if use_ai:
        ai = MiniAI.load(mp)
        X  = df_feat[FCOLS].values
        # Проверяем совместимость модели (могла быть обучена с другим набором фич)
        if hasattr(ai, 'x_mean') and ai.x_mean is not None and len(ai.x_mean) != X.shape[1]:
            log(f'  ⚠️  Модель несовместима ({len(ai.x_mean)} фич vs {X.shape[1]}) — fallback')
            use_ai = False
        else:
            preds = ai.predict(X)
            log(f'  ✅ AI — {len(preds)} предсказаний')
    if not use_ai:
        known = df[df['TOTAL'] > 0]
        median_ratio = float((known['TOTAL'] / known['GCSP']).median()) if len(known) > 0 else 1.0
        preds = (df['GCSP'] * df['QTY'].clip(lower=1)).values.copy()
        zero_qty_mask = df['QTY'].values == 0.0
        preds[zero_qty_mask] = df['GCSP'].values[zero_qty_mask] * median_ratio
        log('  ⚠️  Нет модели — fallback')

    ws       = wb[sheet_name]
    all_rows = list(ws.iter_rows())
    cur_tot  = cur_qty = None

    for row_cells in all_rows:
        vals = [c.value for c in row_cells]
        if _is_header(vals):
            candidates = [j for j, v in enumerate(vals)
                          if v is not None and 'total' in str(v).lower()
                          and 'sub' not in str(v).lower()]
            cur_tot = (candidates[-1] + 1) if candidates else None
            qty_j   = _col_index(vals, 'qty', 'quantity')
            cur_qty = (qty_j + 1) if qty_j is not None else None
            continue
        if cur_tot is None:
            continue
        if cur_tot <= len(row_cells):
            tcell = row_cells[cur_tot - 1]
            if isinstance(tcell.value, str) and tcell.value.startswith('='):
                fval_up = tcell.value.upper()
                if not any(kw in fval_up for kw in ('SUMIF', 'SUMIFS', 'SUM(', 'COUNT', 'AVERAGE')):
                    _write_cell(tcell, 0)

    row_pred_map = {int(rec['row_index']): {
        'pred': float(preds[idx_pos]),
        'gcsp': float(rec['GCSP']),
        'qty':  float(rec['QTY']),
        'qty_col':   int(rec['qty_col']),
        'total_col': int(rec['total_col']),
    } for idx_pos, (_, rec) in enumerate(df.iterrows())}

    written = qty_written = 0
    for row_cells in all_rows:
        row_idx = row_cells[0].row - 1
        if row_idx not in row_pred_map:
            continue
        meta      = row_pred_map[row_idx]
        tot_col_1 = meta['total_col'] + 1
        qty_col_1 = meta['qty_col']   + 1
        if tot_col_1 <= len(row_cells):
            _write_cell(row_cells[tot_col_1 - 1], meta['pred'])
            written += 1
        if meta['qty'] == 0.0 and qty_col_1 <= len(row_cells):
            computed_qty = max(0.0, meta['pred'] / meta['gcsp']) if meta['gcsp'] > 0 else 0.0
            qcell = row_cells[qty_col_1 - 1]
            if _to_float(qcell.value) in (None, 0.0):
                qcell.value = int(round(computed_qty))
                qcell.number_format = '0'
                qty_written += 1

    log(f'  ✅ {written} TOTAL, {qty_written} QTY ячеек записано')
    all_results.append(dict(sheet=sheet_name, rows=len(df), written=written,
                            method='AI' if use_ai else 'fallback',
                            best_wire_row=None, best_nn_score=None, best_real_score=None))


# ─── 4. SIMPLE QTY SHEETS — QTY=1 везде ─────────────────────────────────────

for sheet_name in SIMPLE_QTY_SHEETS:
    log()
    log(f'  Sheet: {sheet_name}  [QTY=1 mode]')

    if sheet_name not in wb.sheetnames:
        log(f'  ⚠️  Лист не найден, пропуск.')
        all_results.append(dict(sheet=sheet_name, rows=0, written=0,
                                method='QTY=1', best_wire_row=None,
                                best_nn_score=None, best_real_score=None))
        continue

    ws = wb[sheet_name]
    all_rows = list(ws.iter_rows())
    cur_qty  = None
    cur_gcsp = None
    qty_written = 0

    for row_cells in all_rows:
        vals = [c.value for c in row_cells]

        # Ищем строку-заголовок по наличию 'qty'/'quantity'
        qty_j = _col_index(vals, 'qty', 'quantity')
        if qty_j is not None:
            cur_qty  = qty_j + 1
            gcsp_j   = _col_index(vals, 'global sec')
            cur_gcsp = gcsp_j   # None если колонки gcsp нет
            continue

        if cur_qty is None:
            continue

        # Если есть gcsp-колонка — фильтруем по gcsp > 0
        # Если нет — ставим QTY=1 любой непустой строке
        if cur_gcsp is not None:
            gcsp_val = _to_float(vals[cur_gcsp]) if cur_gcsp < len(vals) else None
            if gcsp_val is None or gcsp_val <= 0:
                continue
        else:
            if not any(v is not None for v in vals):
                continue

        if cur_qty <= len(row_cells):
            qcell = row_cells[cur_qty - 1]
            existing = _to_float(qcell.value)
            if qcell.value is None or existing == 0.0:
                qcell.value = 1
                qcell.number_format = '0'
                qty_written += 1

    log(f'  ✅ QTY=1 проставлено в {qty_written} ячейках')
    all_results.append(dict(sheet=sheet_name, rows=qty_written, written=qty_written,
                            method='QTY=1', best_wire_row=None,
                            best_nn_score=None, best_real_score=None))


# ─── СОХРАНЕНИЕ И ИТОГ ────────────────────────────────────────────────────────

combined_out = os.path.join(OUTPUT_DIR, 'RESULT_ALL_SHEETS.xlsx')
wb.save(combined_out)

session_elapsed = (datetime.datetime.now() - session_start).total_seconds()

log()
log('  ' + '═' * W)
log(f'  {"📋  SESSION SUMMARY":^{W}}')
log('  ' + '═' * W)
log()

for r in all_results:
    bw = (f'row {r["best_wire_row"]}  NN={r["best_nn_score"]:.2f}  real={r["best_real_score"]}'
          if r['best_wire_row'] is not None else '—')
    log(f'  {r["sheet"]:<28}  method={r["method"]:10}  best={bw}')

# ─── ЛУЧШИЙ ПРОВОД — финальный баннер ────────────────────────────────────────
log()
log('  ' + '═' * W)
log(f'  {"🏆  ЛУЧШИЙ ПРОВОД  🏆":^{W}}')
log('  ' + '═' * W)
log(f'  {"Excel-строка":<22}: {best_excel_row}')
log(f'  {"Группа машины":<22}: {best_row["machine_group"]}')
log(f'  {"Gage":<22}: [{best_row["min_gage"]:.2f} – {best_row["max_gage"]:.2f}]  (mid={best_mid_g:.2f})')
log(f'  {"Length":<22}: [{best_row["min_len"]:.0f} – {best_row["max_len"]:.0f}]  (mid={best_mid_l:.0f})')
log(f'  {"GCSP":<22}: {best_row["gcsp"]:.4f}  сек/шт')
log()
log(f'  {"NN predicted score":<22}: {nn_scores[best_idx]:.2f}')
log(f'  {"Реальный score":<22}: {int(real_scores[best_idx])}  из {total_ops} операций')
log()
log(f'  Детализация по пространствам:')
lp_total   = len(filters_dict["LEAD PREP"])
lpfa_total = len(filters_dict["LEAD PREP FA"])
hv_total   = len(filters_dict["High Voltage"])
log(f'    {"LEAD PREP":<20}: {best_lp:3d}  из {lp_total:2d}  операций  ({100*best_lp//max(1,lp_total)}%)')
log(f'    {"LEAD PREP FA":<20}: {best_lpfa:3d}  из {lpfa_total:2d}  операций  ({100*best_lpfa//max(1,lpfa_total)}%)')
log(f'    {"High Voltage":<20}: {best_hv:3d}  из {hv_total:2d}  операций  ({100*best_hv//max(1,hv_total)}%)')
log()
log(f'  📁  Результат сохранён: {combined_out}')
log(f'  ⏱  Общее время:         {session_elapsed:.1f} сек')
log('  ' + '═' * W)

_log_file.close()