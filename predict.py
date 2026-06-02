import os
import datetime
import numpy as np
from openpyxl import load_workbook

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
    create_features_augmented,
    load_and_prepare_total_gcsd,
    create_features_total_gcsd,
    _is_header,
    _to_float,
    _col_index,
    _encode_machine_group,
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


def _pred_stats(preds, label='Predictions'):
    if len(preds) == 0:
        return f'  {label}: (empty)'
    return (f'  {label}: n={len(preds)}'
            f'  mean={np.mean(preds):.4f}'
            f'  std={np.std(preds):.4f}'
            f'  min={np.min(preds):.4f}'
            f'  max={np.max(preds):.4f}')


def _pass_fail_inline(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return y_pred <= y_true


# ─────────────────────────────────────────────────────────────────────────────

session_start = datetime.datetime.now()
W = 70

log()
log('  ' + '═' * W)
log(f'  {"🔮  PREDICTION SESSION (Wire-by-Wire Concept)":^{W}}')
log(f'  {"Started: " + session_start.strftime("%Y-%m-%d  %H:%M:%S"):^{W}}')
log(f'  {"Input:  " + FILE_PATH:^{W}}')
log(f'  {"Output: results/RESULT_ALL_SHEETS.xlsx":^{W}}')
log('  ' + '═' * W)

all_results = []
wb = load_workbook(FILE_PATH)


# ─── 1. CUTTING (Wire-by-Wire) ─────────────────────────────────────────────────

log()
log('  ' + '=' * W)
log('  Sheet: CUTTING  (Wire-by-Wire Prediction)')
log('  ' + '=' * W)
log()

df_cutting = load_cutting_sheet(FILE_PATH, 'CUTTING')

mp     = model_path('CUTTING')
use_ai = os.path.exists(mp)

if use_ai:
    ai_cut = MiniAI.load(mp)

    # Получаем group_mapping из модели (если сохранён) или пересчитываем
    if hasattr(ai_cut, 'group_mapping') and ai_cut.group_mapping:
        gmap   = ai_cut.group_mapping
        ggmean = ai_cut.group_global_mean or 0.0
    else:
        _, gmap, ggmean = _encode_machine_group(df_cutting)

    df_feat = create_cutting_features(df_cutting, gmap, ggmean)
    X       = df_feat[ai_cut.feature_cols].values
    preds   = ai_cut.predict(X)
    log(f'  ✅ AI модель загружена — {len(preds)} предсказаний')
    if ai_cut.history:
        m = ai_cut.history.get('metrics_all', {})
        log(f'  Качество модели (из обучения):')
        log(f'    MAE={m.get("mae",0):.4f}  RMSE={m.get("rmse",0):.4f}'
            f'  R²={m.get("r2",0):.4f}  MAPE={m.get("mape",0):.2f}%')
else:
    preds = df_cutting['gcsp'].values.copy()
    log(f'  ⚠️  Нет модели — fallback (pred = table GCSP)')

table_gcsp = df_cutting['gcsp'].values
passed     = _pass_fail_inline(table_gcsp, preds)

log()
log(f'  ── Wire-by-Wire Pass/Fail Report ──')
log(f'  Всего строк    : {len(preds)}')
log(f'  Pass (pred ≤ ref): {passed.sum():4d}  ({passed.mean()*100:.1f}%)')
log(f'  Fail (pred > ref): {(~passed).sum():4d}  ({(~passed).mean()*100:.1f}%)')
log()

log('  Детали (первые 10 строк):')
log(f'  {"Excel":>6} {"MinG":>6} {"MaxG":>6} {"MinL":>7} {"MaxL":>7}'
    f' {"GCSP_ref":>9} {"GCSP_pred":>9} {"Pass":>6}')
log('  ' + '-' * 68)
for idx in range(min(10, len(df_cutting))):
    row = df_cutting.iloc[idx]
    log(f'  {int(row["excel_row"]):6d} {row["min_gage"]:6.2f} {row["max_gage"]:6.2f}'
        f' {row["min_len"]:7.0f} {row["max_len"]:7.0f}'
        f' {row["gcsp"]:9.4f} {preds[idx]:9.4f}'
        f' {"✅" if passed[idx] else "❌":>6}')
if len(df_cutting) > 10:
    log('  ...')
    for idx in range(max(10, len(df_cutting) - 3), len(df_cutting)):
        row = df_cutting.iloc[idx]
        log(f'  {int(row["excel_row"]):6d} {row["min_gage"]:6.2f} {row["max_gage"]:6.2f}'
            f' {row["min_len"]:7.0f} {row["max_len"]:7.0f}'
            f' {row["gcsp"]:9.4f} {preds[idx]:9.4f}'
            f' {"✅" if passed[idx] else "❌":>6}')

# Записываем в Excel
# BUG FIX: QTY=1 ставим ТОЛЬКО для строк где QTY был 0/None,
# чтобы не разрушать оригинальные QTY
ws_cut = wb['CUTTING']
written_cut = 0

for idx, (_, row_data) in enumerate(df_cutting.iterrows()):
    excel_row = int(row_data['excel_row'])
    pred_gcsp = float(preds[idx])

    # QTY: ставим 1 только если оригинальный QTY=0 (слот-строка)
    orig_qty = row_data['qty']
    if orig_qty == 0.0:
        ws_cut.cell(row=excel_row, column=9).value = int(1)
        ws_cut.cell(row=excel_row, column=9).number_format = '0'

    # TOTAL TIME = pred_gcsp (в J, 10-й столбец)
    _write_cell(ws_cut.cell(row=excel_row, column=10), pred_gcsp)
    written_cut += 1

log()
log(f'  ✅ Записано {written_cut} строк (pred_GCSP в J, QTY=1 только для пустых слотов)')

all_results.append(dict(
    sheet='CUTTING', rows=len(df_cutting), written=written_cut,
    method='AI' if use_ai else 'fallback',
    pass_pct=float(passed.mean() * 100),
    mean_p=float(np.mean(preds)), std_p=float(np.std(preds)),
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
    log(f'  ✅ AI модель загружена — {len(preds)} предсказаний')
else:
    preds = (df_gcsd['GCSD'] * df_gcsd['ADJ.']).clip(lower=0).values
    log(f'  ⚠️  Нет модели — fallback GCSD×ADJ')

known_mask = df_gcsd['PLANT STANDARD'].notna().values
if known_mask.sum() > 0:
    known_true = df_gcsd.loc[known_mask, 'PLANT STANDARD'].values
    known_pred = preds[known_mask]
    passed_g   = _pass_fail_inline(known_true, known_pred)
    log(f'  Pass (pred ≤ ref): {passed_g.sum()}/{known_mask.sum()} '
        f'({passed_g.mean()*100:.1f}%)')

ws_g   = wb['TOTAL values GCSD']
E_COL, M_COL = 5, 13

for col_1 in set(int(df_gcsd.iloc[i]['plant_col']) + 1 for i in range(len(df_gcsd))):
    for row in ws_g.iter_rows(min_col=col_1, max_col=col_1):
        cell = row[0]
        if isinstance(cell.value, str) and cell.value.startswith('='):
            fval_up = cell.value.upper()
            if not any(kw in fval_up for kw in ('SUMIF', 'SUMIFS', 'SUM(', 'COUNT', 'AVERAGE')):
                _write_cell(cell, 0)

for stale_row in (24, 25, 26):
    cell = ws_g.cell(row=stale_row, column=E_COL)
    if cell.value is not None and not (isinstance(cell.value, str) and 'SUM' in str(cell.value).upper()):
        cell.value = None

written_gcsd = 0
for i, pred in enumerate(preds):
    row_num = int(df_gcsd.iloc[i]['row_index']) + 1
    col_1   = int(df_gcsd.iloc[i]['plant_col']) + 1
    _write_cell(ws_g.cell(row=row_num, column=col_1), pred)
    written_gcsd += 1

ws_g.cell(row=13, column=E_COL).value = '=SUM(E6:E12)'
ws_g.cell(row=23, column=E_COL).value = '=SUM(E19:E22)'
ws_g.cell(row=27, column=M_COL).value = '=SUM(M19:M26)'

log(f'  ✅ Записано {written_gcsd} ячеек, SUM формулы E13/E23/M27 установлены')

all_results.append(dict(
    sheet='TOTAL values GCSD', rows=len(df_gcsd), written=written_gcsd,
    method='AI' if use_ai else 'fallback',
    pass_pct=float(passed_g.mean()*100) if known_mask.sum() > 0 else 0.0,
    mean_p=float(np.mean(preds)), std_p=float(np.std(preds)),
))


# ─── 3. GENERIC SHEETS ────────────────────────────────────────────────────────

for sheet_name in GENERIC_SHEETS:
    log()
    log('  ' + '=' * W)
    log(f'  Sheet: {sheet_name}')
    log('  ' + '=' * W)

    df, _ = load_sheet_dynamic(FILE_PATH, sheet_name)
    if len(df) == 0:
        log('  ⚠️  Нет данных — пропуск.')
        all_results.append(dict(sheet=sheet_name, rows=0, written=0,
                                method='—', pass_pct=0.0, mean_p=0.0, std_p=0.0))
        continue

    med_min_gage = float(df_cutting['min_gage'].median())
    med_max_gage = float(df_cutting['max_gage'].median())
    med_min_len  = float(df_cutting['min_len'].median())
    med_max_len  = float(df_cutting['max_len'].median())
    df = df.copy()
    df['min_gage'] = med_min_gage
    df['max_gage'] = med_max_gage
    df['min_len']  = med_min_len
    df['max_len']  = med_max_len

    df_feat = create_features_augmented(df)
    mp      = model_path(sheet_name)
    use_ai  = os.path.exists(mp)

    if use_ai:
        ai    = MiniAI.load(mp)
        X     = df_feat[ai.feature_cols].values
        preds = ai.predict(X)
        log(f'  ✅ AI модель загружена — {len(preds)} предсказаний')
    else:
        known = df[df['TOTAL'] > 0]
        median_ratio = float((known['TOTAL'] / known['GCSP']).median()) if len(known) > 0 else 1.0
        preds = (df['GCSP'] * df['QTY'].clip(lower=1)).values.copy()
        zero_qty_mask = df['QTY'].values == 0.0
        preds[zero_qty_mask] = df['GCSP'].values[zero_qty_mask] * median_ratio
        log(f'  ⚠️  Нет модели — fallback')

    known_vals  = df['TOTAL'].values
    known_mask2 = known_vals > 0
    if known_mask2.sum() > 0:
        passed2 = _pass_fail_inline(known_vals[known_mask2], preds[known_mask2])
        log(f'  Pass (pred ≤ ref): {passed2.sum()}/{known_mask2.sum()} '
            f'({passed2.mean()*100:.1f}%)')
        pass_pct2 = float(passed2.mean() * 100)
    else:
        pass_pct2 = 0.0

    ws       = wb[sheet_name]
    all_rows = list(ws.iter_rows())
    cur_tot  = cur_qty = None
    cleared  = 0

    for row_cells in all_rows:
        vals = [c.value for c in row_cells]
        if _is_header(vals):
            candidates = [j for j, v in enumerate(vals)
                          if v is not None
                          and 'total' in str(v).lower()
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
                if not any(kw in fval_up for kw in ('SUMIF','SUMIFS','SUM(','COUNT','AVERAGE')):
                    _write_cell(tcell, 0)
                    cleared += 1
        if cur_qty and cur_qty <= len(row_cells):
            qcell = row_cells[cur_qty - 1]
            if isinstance(qcell.value, str) and qcell.value.startswith('='):
                qcell.value = int(0)
                qcell.number_format = '0'

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
        # BUG FIX: QTY заполняем только если реально 0 (слот)
        if meta['qty'] == 0.0 and qty_col_1 <= len(row_cells):
            computed_qty = max(0.0, meta['pred'] / meta['gcsp']) if meta['gcsp'] > 0 else 0.0
            qcell = row_cells[qty_col_1 - 1]
            if _to_float(qcell.value) in (None, 0.0):
                qcell.value = int(round(computed_qty))
                qcell.number_format = '0'
                qty_written += 1

    log(f'  ✅ Записано {written} TOTAL ячеек, {qty_written} QTY ячеек')

    all_results.append(dict(
        sheet=sheet_name, rows=len(df), written=written,
        method='AI' if use_ai else 'fallback',
        pass_pct=pass_pct2,
        mean_p=float(np.mean(preds)), std_p=float(np.std(preds)),
    ))


# ─── СОХРАНЕНИЕ И ИТОГ ────────────────────────────────────────────────────────

combined_out = os.path.join(OUTPUT_DIR, 'RESULT_ALL_SHEETS.xlsx')
wb.save(combined_out)

session_end     = datetime.datetime.now()
session_elapsed = (session_end - session_start).total_seconds()

log()
log('  ' + '═' * W)
log(f'  {"📋  PREDICTION SESSION SUMMARY":^{W}}')
log('  ' + '═' * W)
log()

col_w = [24, 6, 8, 10, 10, 10, 10]
total_w = sum(col_w) + 16
header = (f'  │  {"Sheet":<{col_w[0]}}'
          f'{"Rows":>{col_w[1]}}'
          f'{"Written":>{col_w[2]}}'
          f'{"Method":>{col_w[3]}}'
          f'{"Pass%":>{col_w[4]}}'
          f'{"Mean":>{col_w[5]}}'
          f'{"Std":>{col_w[6]}}  │')

log('  ┌' + '─' * total_w + '┐')
log(header)
log('  ├' + '─' * total_w + '┤')
for r in all_results:
    log(f'  │  {r["sheet"]:<{col_w[0]}}'
        f'{r["rows"]:>{col_w[1]}}'
        f'{r["written"]:>{col_w[2]}}'
        f'{r["method"]:>{col_w[3]}}'
        f'{r["pass_pct"]:>{col_w[4]}.1f}'
        f'{r["mean_p"]:>{col_w[5]}.3f}'
        f'{r["std_p"]:>{col_w[6]}.3f}  │')
log('  └' + '─' * total_w + '┘')

log()
log(f'  Записано всего ячеек   : {sum(r["written"] for r in all_results)}')
log(f'  Выходной файл          : {combined_out}')
log(f'  ⏱  Время               : {session_elapsed:.1f} сек')
log(f'  📁  Лог                 : logs/predict.log')
log('  ' + '═' * W)
log()

_log_file.close()