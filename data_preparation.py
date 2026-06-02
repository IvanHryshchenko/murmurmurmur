"""
data_preparation.py
-------------------
Парсинг всех листов K0_old.xlsx.

Ключевое изменение (концепт заказчика):
  Лист CUTTING — строки I3:I681 — это "TOTAL Qty Leads".
  Каждая строка — самостоятельная операция с параметрами:
    Min Gage, Max Gage, Min Length, Max Length, Global Sec/Pc (GCSP).
  
  Нейросеть учится ПРЕДСКАЗЫВАТЬ Global Sec/Pc (время на операцию)
  по параметрам Min/Max Gage и Min/Max Length.
  
  При обучении/предсказании QTY=1 подставляется в каждую строку по очереди
  (I3=1, I4=1, ..., I681=1), чтобы активировать строку и получить
  предсказание TOTAL TIME = GCSP * QTY = GCSP * 1 = GCSP.

  Цель: предсказанное GCSP должно совпасть или быть чуть меньше эталонного
  табличного значения.
"""

import re
import numpy as np
import pandas as pd
from openpyxl import load_workbook


# ─── ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ─────────────────────────────────────────────────

def _to_float(v):
    try:
        f = float(v)
        return None if (np.isnan(f) or np.isinf(f)) else f
    except (TypeError, ValueError):
        return None


def _is_ne(row_vals):
    """Пропускаем строки 'не учитываю'."""
    for v in row_vals:
        if v is not None and 'не учитываю' in str(v).lower():
            return True
    return False


def _is_header(row_vals):
    """Строка является заголовком, если содержит 'global sec'."""
    for v in row_vals:
        if v is not None and 'global sec' in str(v).lower():
            return True
    return False


def _col_index(row_vals, *keywords):
    for kw in keywords:
        for j, v in enumerate(row_vals):
            if v is not None and kw.lower() in str(v).lower():
                return j
    return None


_GCSD_SKIP_LABELS = {'total', 'per harness', 'part number', 'total time',
                     'wires number', '(min)', '%', 'summary', 'high voltage'}


def _gcsd_skip(label):
    s = label.lower().strip()
    if not s or len(s) < 2:
        return True
    for kw in _GCSD_SKIP_LABELS:
        if kw in s:
            return True
    return False


# ─── CUTTING: ПОСТРОЧНЫЙ РАЗБОР С ПАРАМЕТРАМИ ПРОВОДА ────────────────────────

def load_cutting_sheet(file_path, sheet_name='CUTTING'):
    """
    Разбирает лист CUTTING.

    Структура листа:
      Row 2 — заголовок: Comments | MachineVendor | Machine |
               Min Gage | Max Gage | Min Length | Max Length |
               Global Sec/Pc | TOTAL Qty Leads | SUB TOTAL | Qty Marked | TOTAL TIME
      Row 3..681 — данные по операциям/проводам

    Возвращает DataFrame со столбцами:
      row_index, min_gage, max_gage, min_len, max_len,
      gcsp (=Global Sec/Pc = эталонное время/шт),
      qty  (=TOTAL Qty Leads, обычно None в шаблоне),
      machine_group (название группы из столбца Comments),
      excel_row (1-based номер строки в Excel)
    """
    wb = load_workbook(file_path, read_only=True, data_only=True)
    ws = wb[sheet_name]
    rows_data = list(ws.iter_rows(min_row=1, values_only=True))

    # Колонки (0-based): D=3 MinGage, E=4 MaxGage, F=5 MinLen, G=6 MaxLen,
    #                    H=7 GCSP, I=8 QTY
    COL_MIN_GAGE = 3
    COL_MAX_GAGE = 4
    COL_MIN_LEN  = 5
    COL_MAX_LEN  = 6
    COL_GCSP     = 7
    COL_QTY      = 8

    records = []
    current_group = 'Unknown'

    # Строки 3..681 (0-based: 2..680)
    for i in range(2, min(681, len(rows_data))):
        row = rows_data[i]

        # Обновляем группу машины, если в столбце A есть текст
        comment_val = row[0] if len(row) > 0 else None
        if comment_val is not None:
            s = str(comment_val).strip()
            if s and len(s) > 2:
                current_group = s[:80]

        def _get(col):
            return row[col] if len(row) > col else None

        gcsp = _to_float(_get(COL_GCSP))
        if gcsp is None or gcsp <= 0:
            continue

        min_gage = _to_float(_get(COL_MIN_GAGE))
        max_gage = _to_float(_get(COL_MAX_GAGE))
        min_len  = _to_float(_get(COL_MIN_LEN))
        max_len  = _to_float(_get(COL_MAX_LEN))
        qty_raw  = _to_float(_get(COL_QTY))

        # Требуем хотя бы одну из пар параметров
        has_gage = (min_gage is not None or max_gage is not None)
        has_len  = (min_len  is not None or max_len  is not None)
        if not (has_gage or has_len):
            continue

        records.append({
            'row_index':     i,
            'excel_row':     i + 1,           # 1-based
            'machine_group': current_group,
            'min_gage':      min_gage if min_gage is not None else 0.0,
            'max_gage':      max_gage if max_gage is not None else 0.0,
            'min_len':       min_len  if min_len  is not None else 0.0,
            'max_len':       max_len  if max_len  is not None else 0.0,
            'gcsp':          gcsp,             # эталон: Global Sec/Pc
            'qty':           qty_raw if qty_raw is not None else 0.0,
        })

    wb.close()
    df = pd.DataFrame(records) if records else pd.DataFrame(columns=[
        'row_index', 'excel_row', 'machine_group',
        'min_gage', 'max_gage', 'min_len', 'max_len', 'gcsp', 'qty'
    ])
    return df


def create_cutting_features(df):
    """
    Инжиниринг признаков для листа CUTTING.

    Входные признаки (что знаем о проводе/операции):
      min_gage, max_gage, min_len, max_len
    Целевая переменная: gcsp (Global Sec/Pc)

    При подстановке QTY=1:  TOTAL_TIME = gcsp * 1 = gcsp
    """
    df = df.copy()

    # Основные диапазоны
    df['gage_range']   = df['max_gage'] - df['min_gage']
    df['len_range']    = df['max_len']  - df['min_len']
    df['mid_gage']     = (df['min_gage'] + df['max_gage']) / 2
    df['mid_len']      = (df['min_len']  + df['max_len'])  / 2

    # Логарифмические трансформации (стабилизируют большие диапазоны длин)
    df['min_gage_log'] = np.log1p(df['min_gage'].clip(lower=0))
    df['max_gage_log'] = np.log1p(df['max_gage'].clip(lower=0))
    df['min_len_log']  = np.log1p(df['min_len'].clip(lower=0))
    df['max_len_log']  = np.log1p(df['max_len'].clip(lower=0))
    df['mid_len_log']  = np.log1p(df['mid_len'].clip(lower=0))
    df['mid_gage_log'] = np.log1p(df['mid_gage'].clip(lower=0))

    # Квадратные корни
    df['max_gage_sqrt'] = np.sqrt(df['max_gage'].clip(lower=0))
    df['max_len_sqrt']  = np.sqrt(df['max_len'].clip(lower=0))
    df['mid_len_sqrt']  = np.sqrt(df['mid_len'].clip(lower=0))

    # Взаимодействие: время резки зависит и от толщины и от длины
    df['gage_x_len']    = df['mid_gage'] * df['mid_len']
    df['gage_x_maxlen'] = df['max_gage'] * df['max_len']

    return df


# Колонки признаков для CUTTING
FEATURE_COLS_CUTTING = [
    'min_gage', 'max_gage', 'min_len', 'max_len',
    'gage_range', 'len_range', 'mid_gage', 'mid_len',
    'min_gage_log', 'max_gage_log', 'min_len_log', 'max_len_log',
    'mid_len_log', 'mid_gage_log',
    'max_gage_sqrt', 'max_len_sqrt', 'mid_len_sqrt',
    'gage_x_len', 'gage_x_maxlen',
]


# ─── TOTAL VALUES GCSD ────────────────────────────────────────────────────────

def load_and_prepare_total_gcsd(file_path, sheet_name='TOTAL values GCSD'):
    xl = pd.ExcelFile(file_path)
    df_original = xl.parse(sheet_name, header=None)
    data = []
    BLOCKS = [
        (1, 2, 3, 4),
        (9, 10, 11, 12),
    ]

    for i in range(len(df_original)):
        row = df_original.iloc[i].values

        for label_col, gcsd_col, adj_col, plant_col in BLOCKS:
            def _c(col, row=row):
                return row[col] if col < len(row) else None

            label_raw = _c(label_col)
            try:
                if pd.isna(label_raw):
                    continue
            except (TypeError, ValueError):
                pass
            label = str(label_raw).strip()
            if _gcsd_skip(label):
                continue

            gcsd  = _to_float(_c(gcsd_col))
            adj   = _to_float(_c(adj_col))
            plant = _to_float(_c(plant_col))

            if gcsd is None or gcsd <= 0:
                continue
            if adj is None or adj <= 0:
                continue

            data.append({
                'row_index':      i,
                'PART NUMBER':    label,
                'GCSD':           gcsd,
                'ADJ.':           adj,
                'PLANT STANDARD': plant,
                'plant_col':      plant_col,
            })

    df_clean = pd.DataFrame(data) if data else pd.DataFrame(
        columns=['row_index', 'PART NUMBER', 'GCSD', 'ADJ.',
                 'PLANT STANDARD', 'plant_col'])
    return df_clean, df_original


def create_features_total_gcsd(df):
    df = df.copy()
    df['GCSD_log']        = np.log1p(df['GCSD'])
    df['ADJ_log']         = np.log1p(df['ADJ.'])
    df['GCSD_x_ADJ']      = df['GCSD'] * df['ADJ.']
    df['GCSD_sqrt']       = np.sqrt(df['GCSD'])
    df['ADJ_sqrt']        = np.sqrt(df['ADJ.'])
    df['PART_LEN']        = df['PART NUMBER'].astype(str).str.len()
    df['PART_DIGITS']     = df['PART NUMBER'].astype(str).str.count(r'\d')
    df['PART_HAS_LETTER'] = df['PART NUMBER'].astype(str).str.contains(r'[A-Za-z]').astype(int)
    df['PLANT STANDARD']  = df['PLANT STANDARD'].fillna(df['GCSD'] * df['ADJ.'])
    return df


FEATURE_COLS_TOTAL_GCSD = [
    'GCSD', 'ADJ.',
    'GCSD_log', 'ADJ_log',
    'GCSD_x_ADJ', 'GCSD_sqrt', 'ADJ_sqrt',
    'PART_LEN', 'PART_DIGITS', 'PART_HAS_LETTER',
]


# ─── GENERIC SHEETS (LEAD PREP, FA, etc.) ────────────────────────────────────

def _detect_cols_from_header(row_vals, prev_gcsp, prev_qty, prev_total):
    new_gcsp = _col_index(row_vals, 'global sec')
    gcsp_col = new_gcsp if new_gcsp is not None else prev_gcsp

    new_qty  = _col_index(row_vals, 'qty', 'quantity')
    qty_col  = new_qty if new_qty is not None else prev_qty

    candidates = [
        j for j, v in enumerate(row_vals)
        if v is not None
        and 'total' in str(v).lower()
        and 'sub' not in str(v).lower()
    ]
    total_col = candidates[-1] if candidates else prev_total

    return gcsp_col, qty_col, total_col


def load_sheet_dynamic(file_path, sheet_name):
    xl = pd.ExcelFile(file_path)
    df_raw = xl.parse(sheet_name, header=None)

    records  = []
    gcsp_col = qty_col = total_col = None

    for i, row in df_raw.iterrows():
        vals = list(row.values)
        if _is_ne(vals):
            continue
        if _is_header(vals):
            gcsp_col, qty_col, total_col = _detect_cols_from_header(
                vals, gcsp_col, qty_col, total_col)
            continue
        if gcsp_col is None or qty_col is None or total_col is None:
            continue

        gcsp  = _to_float(vals[gcsp_col]  if gcsp_col  < len(vals) else None)
        total = _to_float(vals[total_col] if total_col < len(vals) else None)

        if gcsp is None or gcsp <= 0:
            continue
        if total is None:
            continue

        qty_raw = vals[qty_col] if qty_col < len(vals) else None
        qty = _to_float(qty_raw)
        if qty is None:
            qty = 0.0

        cat = ''
        for v in vals:
            if v is not None:
                f = _to_float(v)
                if f is None:
                    s = str(v).strip()
                    if s and s.lower() != 'nan':
                        cat = s
                        break

        records.append({
            'row_index': i,
            'GCSP':      gcsp,
            'QTY':       qty,
            'TOTAL':     total,
            'CATEGORY':  cat,
            'qty_col':   qty_col,
            'total_col': total_col,
        })

    return pd.DataFrame(records), df_raw


def create_features(df):
    df = df.copy()
    df['GCSP_log']        = np.log1p(df['GCSP'])
    df['QTY_log']         = np.log1p(df['QTY'].clip(lower=0))
    df['GCSP_x_QTY']      = df['GCSP'] * df['QTY']
    df['GCSP_sqrt']       = np.sqrt(df['GCSP'].clip(lower=0))
    df['QTY_sqrt']        = np.sqrt(df['QTY'].clip(lower=0))
    df['CAT_LEN']         = df['CATEGORY'].str.len().fillna(0)
    df['CAT_DIGITS']      = df['CATEGORY'].str.count(r'\d').fillna(0)
    df['CAT_HAS_LETTERS'] = df['CATEGORY'].str.contains(r'[A-Za-z]', na=False).astype(int)
    return df


FEATURE_COLS = [
    'GCSP', 'QTY',
    'GCSP_log', 'QTY_log',
    'GCSP_x_QTY', 'GCSP_sqrt', 'QTY_sqrt',
    'CAT_LEN', 'CAT_DIGITS', 'CAT_HAS_LETTERS',
]

GENERIC_SHEETS = [
    'LEAD PREP',
    'LEAD PREP FA',
    'High Voltage',
    'FA Conns and wires',
    'FA Taping',
    'FA Miscellaneos',
]


# ─── CROSS-SHEET AUGMENTATION (CUTTING → GENERIC) ────────────────────────────

def build_augmented_dataset(file_path, sheet_name,
                            df_cutting,
                            use_zero_qty_rows=True):
    """
    Строим расширенный обучающий датасет для generic-листа.

    Алгоритм (концепт Кирилла Шилохвостова):
      Для каждой строки i из CUTTING (строки 3..681):
        1. Берём: min_gage_i, max_gage_i, min_len_i, max_len_i, gcsp_i
        2. Загружаем generic-лист (LEAD PREP и т.д.)
        3. Для каждой строки j в generic-листе, где QTY=0/None:
              - Подставляем QTY=1
              - Если в строке j нет своих min/max gage/len — подставляем из строки i CUTTING
              - TOTAL = GCSP_j * QTY = GCSP_j * 1 = GCSP_j  (GCSP берём из листа)
              - Записываем как обучающую точку
        4. Строки j с уже заполненным QTY > 0 — берём как есть (реальные данные)

    Признаки для каждой точки:
        GCSP (из generic-листа), QTY=1,
        min_gage, max_gage, min_len, max_len (из CUTTING если нет своих),
        + производные фичи

    Целевая переменная: TOTAL = GCSP * QTY

    Возвращает DataFrame со всеми обучающими точками.
    Размер: len(df_cutting) * len(zero_qty_rows_in_sheet) + real_rows
    """
    # 1. Загружаем generic-лист один раз
    df_sheet, _ = load_sheet_dynamic(file_path, sheet_name)

    if df_sheet.empty:
        return pd.DataFrame()

    # Разделяем на строки с реальными QTY и строки-"слоты" (QTY=0/None)
    real_rows = df_sheet[df_sheet['TOTAL'] > 0].copy()
    slot_rows = df_sheet[df_sheet['TOTAL'] <= 0].copy() if use_zero_qty_rows else pd.DataFrame()

    augmented_records = []

    # 2. Реальные строки идут в обучение без изменений (wire params неизвестны → 0)
    for _, r in real_rows.iterrows():
        augmented_records.append({
            'GCSP':      r['GCSP'],
            'QTY':       r['QTY'],
            'TOTAL':     r['TOTAL'],
            'CATEGORY':  r['CATEGORY'],
            'min_gage':  0.0,
            'max_gage':  0.0,
            'min_len':   0.0,
            'max_len':   0.0,
            'source':    'real',
        })

    # 3. Для каждой строки i из CUTTING — активируем слоты с QTY=1
    if not slot_rows.empty:
        for _, cutting_row in df_cutting.iterrows():
            c_min_gage = float(cutting_row['min_gage'])
            c_max_gage = float(cutting_row['max_gage'])
            c_min_len  = float(cutting_row['min_len'])
            c_max_len  = float(cutting_row['max_len'])
            # gcsp_cutting = cutting_row['gcsp']  # не используем напрямую

            for _, slot in slot_rows.iterrows():
                gcsp_j = float(slot['GCSP'])
                total_j = gcsp_j * 1.0  # QTY = 1

                augmented_records.append({
                    'GCSP':      gcsp_j,
                    'QTY':       1.0,
                    'TOTAL':     total_j,
                    'CATEGORY':  slot['CATEGORY'],
                    'min_gage':  c_min_gage,
                    'max_gage':  c_max_gage,
                    'min_len':   c_min_len,
                    'max_len':   c_max_len,
                    'source':    'augmented',
                })

    if not augmented_records:
        return pd.DataFrame()

    df_aug = pd.DataFrame(augmented_records)
    return df_aug


def create_features_augmented(df):
    """
    Инжиниринг признаков для расширенного датасета.
    Объединяет признаки generic-листа + wire-параметры из CUTTING.
    """
    df = df.copy()

    # Стандартные признаки generic-листа
    df['GCSP_log']        = np.log1p(df['GCSP'])
    df['QTY_log']         = np.log1p(df['QTY'].clip(lower=0))
    df['GCSP_x_QTY']      = df['GCSP'] * df['QTY']
    df['GCSP_sqrt']       = np.sqrt(df['GCSP'].clip(lower=0))
    df['QTY_sqrt']        = np.sqrt(df['QTY'].clip(lower=0))
    df['CAT_LEN']         = df['CATEGORY'].astype(str).str.len().fillna(0)
    df['CAT_DIGITS']      = df['CATEGORY'].astype(str).str.count(r'\d').fillna(0)
    df['CAT_HAS_LETTERS'] = df['CATEGORY'].astype(str).str.contains(r'[A-Za-z]', na=False).astype(int)

    # Wire-параметры из CUTTING
    df['gage_range']    = df['max_gage'] - df['min_gage']
    df['len_range']     = df['max_len']  - df['min_len']
    df['mid_gage']      = (df['min_gage'] + df['max_gage']) / 2.0
    df['mid_len']       = (df['min_len']  + df['max_len'])  / 2.0
    df['min_gage_log']  = np.log1p(df['min_gage'].clip(lower=0))
    df['max_gage_log']  = np.log1p(df['max_gage'].clip(lower=0))
    df['min_len_log']   = np.log1p(df['min_len'].clip(lower=0))
    df['max_len_log']   = np.log1p(df['max_len'].clip(lower=0))
    df['mid_len_log']   = np.log1p(df['mid_len'].clip(lower=0))
    df['mid_gage_log']  = np.log1p(df['mid_gage'].clip(lower=0))
    df['max_gage_sqrt'] = np.sqrt(df['max_gage'].clip(lower=0))
    df['max_len_sqrt']  = np.sqrt(df['max_len'].clip(lower=0))

    return df


# Расширенный список признаков (generic + wire из CUTTING)
FEATURE_COLS_AUGMENTED = [
    # Стандартные generic
    'GCSP', 'QTY',
    'GCSP_log', 'QTY_log',
    'GCSP_x_QTY', 'GCSP_sqrt', 'QTY_sqrt',
    'CAT_LEN', 'CAT_DIGITS', 'CAT_HAS_LETTERS',
    # Wire-параметры из CUTTING
    'min_gage', 'max_gage', 'min_len', 'max_len',
    'gage_range', 'len_range', 'mid_gage', 'mid_len',
    'min_gage_log', 'max_gage_log', 'min_len_log', 'max_len_log',
    'mid_len_log', 'mid_gage_log',
    'max_gage_sqrt', 'max_len_sqrt',
]