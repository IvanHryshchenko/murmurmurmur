import os
import datetime
import numpy as np
import pandas as pd

from data_preparation import (
    load_cutting_sheet,
    load_all_filters,
    compute_detailed_score,
    create_cutting_features,
    _encode_machine_group,
    FEATURE_COLS_CUTTING,
)
from model import MiniAI

FILE_PATH  = 'K0_old.xlsx'
MODELS_DIR = 'models'
OUTPUT_DIR = 'results'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs('logs', exist_ok=True)

_log_file = open(os.path.join('logs', 'evaluate.log'), 'a', encoding='utf-8')

W = 70


def log(text=''):
    print(text)
    _log_file.write(text + '\n')
    _log_file.flush()


def bar(value, max_value, width=30, char='█'):
    filled = int(round(value / max(max_value, 1e-9) * width))
    return char * filled + '░' * (width - filled)


def rank_of(idx, score_arr):
    """Место провода idx в рейтинге по убыванию score (1 = лучший)."""
    sorted_idx = np.argsort(score_arr)[::-1]
    pos = np.where(sorted_idx == idx)[0]
    return int(pos[0]) + 1 if len(pos) > 0 else len(score_arr)


# ═══════════════════════════════════════════════════════════════════════════════

session_start = datetime.datetime.now()

log()
log('  ' + '═' * W)
log(f'  {"🔬  WIRE SELECTION — Algorithm vs Neural Network":^{W}}')
log(f'  {"Started: " + session_start.strftime("%Y-%m-%d  %H:%M:%S"):^{W}}')
log('  ' + '═' * W)


# ─── 1. Загрузка данных ───────────────────────────────────────────────────────

log()
log('  Загрузка CUTTING...')
df = load_cutting_sheet(FILE_PATH, 'CUTTING')
log(f'  Проводов: {len(df)}')

log()
log('  Загрузка фильтров операций...')
filters_dict = load_all_filters(FILE_PATH)
lp_total   = len(filters_dict['LEAD PREP'])
lpfa_total = len(filters_dict['LEAD PREP FA'])
hv_total   = len(filters_dict['High Voltage'])
total_ops  = lp_total + lpfa_total + hv_total
log(f'  Итого операций: {total_ops}  (LP={lp_total}  LPFA={lpfa_total}  HV={hv_total})')


# ─── 2. Реальный score для каждого провода ────────────────────────────────────

log()
log('  Вычисление реального score для всех проводов...')

scores_total = []
scores_lp    = []
scores_lpfa  = []
scores_hv    = []

for _, w in df.iterrows():
    mid_g = (w['min_gage'] + w['max_gage']) / 2.0
    mid_l = (w['min_len']  + w['max_len'])  / 2.0
    total, s_lp, s_lpfa, s_hv = compute_detailed_score(mid_g, mid_l, filters_dict)
    scores_total.append(total)
    scores_lp.append(s_lp)
    scores_lpfa.append(s_lpfa)
    scores_hv.append(s_hv)

real_score  = np.array(scores_total, dtype=float)
real_lp     = np.array(scores_lp,    dtype=float)
real_lpfa   = np.array(scores_lpfa,  dtype=float)
real_hv     = np.array(scores_hv,    dtype=float)
gcsp_arr    = df['gcsp'].values
gage_range  = df['max_gage'].values - df['min_gage'].values
len_range   = df['max_len'].values  - df['min_len'].values

log(f'  Score: min={real_score.min():.0f}  max={real_score.max():.0f}  '
    f'mean={real_score.mean():.1f}  std={real_score.std():.1f}')


# ─── 3. Детерминированные алгоритмы ───────────────────────────────────────────

def algo_select(ranking_arr):
    """Выбирает индекс провода с максимальным значением ranking_arr."""
    return int(np.argmax(ranking_arr))


algorithms = {
    'A1_MAX_SCORE':     real_score,
    'A2_MAX_SCORE_FAST': real_score * 10000.0 - gcsp_arr,       # как в NN-ranking
    'A3_MAX_LP':        real_lp,
    'A4_MAX_LPFA':      real_lpfa,
    'A5_MAX_HV':        real_hv,
    'A6_MIN_GCSP':      -gcsp_arr,                              # минимальный GCSP
    'A7_BALANCED':      real_score / np.where(gcsp_arr > 0, gcsp_arr, 1.0),
    'A8_WIDE_GAGE':     gage_range,
}

algo_results = {}
for name, ranking in algorithms.items():
    idx = algo_select(ranking)
    row = df.iloc[idx]
    mid_g = (row['min_gage'] + row['max_gage']) / 2
    mid_l = (row['min_len']  + row['max_len'])  / 2
    algo_results[name] = {
        'idx':        idx,
        'excel_row':  int(row['excel_row']),
        'score':      int(real_score[idx]),
        'lp':         int(real_lp[idx]),
        'lpfa':       int(real_lpfa[idx]),
        'hv':         int(real_hv[idx]),
        'gcsp':       float(row['gcsp']),
        'min_gage':   float(row['min_gage']),
        'max_gage':   float(row['max_gage']),
        'min_len':    float(row['min_len']),
        'max_len':    float(row['max_len']),
        'group':      row['machine_group'],
        # топ-5 индексов по этому алгоритму
        'top5_idx':   set(np.argsort(ranking)[::-1][:5].tolist()),
    }


# ─── 4. NN-предсказание ───────────────────────────────────────────────────────

mp = os.path.join(MODELS_DIR, 'CUTTING', 'model.pkl')
nn_available = os.path.exists(mp)

if nn_available:
    log()
    log('  Загрузка нейросети...')
    ai = MiniAI.load(mp)

    if hasattr(ai, 'group_mapping') and ai.group_mapping:
        gmap, ggmean = ai.group_mapping, ai.group_global_mean or 0.0
    else:
        _, gmap, ggmean = _encode_machine_group(df)

    df_feat  = create_cutting_features(df, gmap, ggmean)
    X        = df_feat[FEATURE_COLS_CUTTING].values
    nn_preds = ai.predict(X)
    nn_rank  = nn_preds * 10000.0 - gcsp_arr
    nn_idx   = int(np.argmax(nn_rank))
    nn_row   = df.iloc[nn_idx]

    nn_result = {
        'idx':       nn_idx,
        'excel_row': int(nn_row['excel_row']),
        'score':     int(real_score[nn_idx]),
        'lp':        int(real_lp[nn_idx]),
        'lpfa':      int(real_lpfa[nn_idx]),
        'hv':        int(real_hv[nn_idx]),
        'gcsp':      float(nn_row['gcsp']),
        'min_gage':  float(nn_row['min_gage']),
        'max_gage':  float(nn_row['max_gage']),
        'min_len':   float(nn_row['min_len']),
        'max_len':   float(nn_row['max_len']),
        'group':     nn_row['machine_group'],
        'nn_score_pred': float(nn_preds[nn_idx]),
        'mae':       float(np.mean(np.abs(nn_preds - real_score))),
        'top5_idx':  set(np.argsort(nn_rank)[::-1][:5].tolist()),
    }
    log(f'  NN выбрал: строка {nn_result["excel_row"]}  '
        f'(predicted={nn_result["nn_score_pred"]:.2f}  real={nn_result["score"]})')
else:
    log()
    log('  ⚠️  Модель CUTTING не найдена — запустите train.py')
    nn_result = None


# ─── 5. Абсолютный оптимум (A1) ───────────────────────────────────────────────

optimum = algo_results['A1_MAX_SCORE']
opt_score = optimum['score']


# ═══════════════════════════════════════════════════════════════════════════════
# ВЫВОД
# ═══════════════════════════════════════════════════════════════════════════════

log()
log('  ' + '═' * W)
log(f'  {"📊  АЛГОРИТМИЧЕСКИЙ АНАЛИЗ":^{W}}')
log('  ' + '═' * W)

# ─── Таблица алгоритмов ───────────────────────────────────────────────────────

log()
log(f'  {"Алгоритм":<22} {"Строка":>6} {"Score":>6} {"LP":>4} {"LPFA":>5} '
    f'{"HV":>4} {"GCSP":>7}  {"Score / макс":^32}')
log('  ' + '─' * W)

for name, r in algo_results.items():
    pct  = r['score'] / opt_score * 100
    bbar = bar(r['score'], opt_score)
    log(f'  {name:<22} {r["excel_row"]:>6} {r["score"]:>6} {r["lp"]:>4} '
        f'{r["lpfa"]:>5} {r["hv"]:>4} {r["gcsp"]:>7.2f}  '
        f'{bbar}  {pct:5.1f}%')

# ─── NN vs алгоритмы ─────────────────────────────────────────────────────────

if nn_result:
    log()
    log('  ' + '═' * W)
    log(f'  {"🤖  НЕЙРОСЕТЬ vs АЛГОРИТМЫ":^{W}}')
    log('  ' + '═' * W)
    log()
    log(f'  NN выбрал: строка {nn_result["excel_row"]}  '
        f'score={nn_result["score"]}  '
        f'(predicted={nn_result["nn_score_pred"]:.2f})')
    log(f'  MAE нейросети на всех проводах: {nn_result["mae"]:.3f}')
    log()

    rank_by_real = rank_of(nn_idx, real_score)
    pct_optimum  = nn_result['score'] / opt_score * 100
    score_gap    = opt_score - nn_result['score']

    log(f'  Место NN-провода в рейтинге по реальному score: '
        f'{rank_by_real} из {len(df)}')
    log(f'  Score NN-провода: {nn_result["score"]}  '
        f'/ Оптимум: {opt_score}  → {pct_optimum:.1f}% от максимума')
    log(f'  Разрыв с оптимумом: {score_gap} операций  '
        f'({score_gap/total_ops*100:.1f}% от всех операций)')
    log()

    # Сравнение с каждым алгоритмом
    log(f'  {"Алгоритм":<22} {"Top1":^6} {"Top5":^6} '
        f'{"Gap score":>10} {"% от оптимума":>14} {"NN rank":>8}')
    log('  ' + '─' * W)

    matches_top1 = 0
    matches_top5 = 0

    for name, r in algo_results.items():
        top1 = '✅' if nn_idx == r['idx'] else '❌'
        top5 = '✅' if nn_idx in r['top5_idx'] else '❌'
        gap  = r['score'] - nn_result['score']   # насколько алго лучше NN
        pct  = nn_result['score'] / r['score'] * 100 if r['score'] > 0 else 0
        rank_in_algo = rank_of(nn_idx, list(algorithms[name]))

        if nn_idx == r['idx']:
            matches_top1 += 1
        if nn_idx in r['top5_idx']:
            matches_top5 += 1

        log(f'  {name:<22} {top1:^6} {top5:^6} '
            f'{gap:>+10} {pct:>13.1f}%  #{rank_in_algo:>5}')

    n_algo = len(algorithms)
    log()
    log(f'  NN совпал с алгоритмом Top-1: {matches_top1}/{n_algo}  '
        f'({matches_top1/n_algo*100:.0f}%)')
    log(f'  NN вошёл в Top-5 алгоритма:  {matches_top5}/{n_algo}  '
        f'({matches_top5/n_algo*100:.0f}%)')

    # ─── Полная сводка провода NN ──────────────────────────────────────────────

    log()
    log('  ' + '═' * W)
    log(f'  {"🏆  ИТОГОВЫЙ ПРОВОД (NN)":^{W}}')
    log('  ' + '═' * W)
    log(f'  {"Excel-строка":<22}: {nn_result["excel_row"]}')
    log(f'  {"Группа":<22}: {nn_result["group"]}')
    log(f'  {"Gage":<22}: [{nn_result["min_gage"]:.2f} – {nn_result["max_gage"]:.2f}]')
    log(f'  {"Length":<22}: [{nn_result["min_len"]:.0f} – {nn_result["max_len"]:.0f}]')
    log(f'  {"GCSP":<22}: {nn_result["gcsp"]:.4f}  сек/шт')
    log()

    pct_lp   = nn_result['lp']   / lp_total   * 100
    pct_lpfa = nn_result['lpfa'] / lpfa_total  * 100
    pct_hv   = nn_result['hv']   / hv_total    * 100 if hv_total > 0 else 0

    log(f'  {"Score (реальный)":<22}: {nn_result["score"]}  из {total_ops}  '
        f'({nn_result["score"]/total_ops*100:.1f}%)')
    log(f'  {"% от оптимума":<22}: {pct_optimum:.1f}%  '
        f'(разрыв: {score_gap} операций)')
    log(f'  {"Место в рейтинге":<22}: #{rank_by_real} из {len(df)}')
    log()
    log(f'  Детализация по пространствам:')
    log(f'    {"LEAD PREP":<20}: {nn_result["lp"]:3d} / {lp_total:2d}  '
        f'  {bar(nn_result["lp"], lp_total, 20)}  {pct_lp:5.1f}%')
    log(f'    {"LEAD PREP FA":<20}: {nn_result["lpfa"]:3d} / {lpfa_total:2d}  '
        f'  {bar(nn_result["lpfa"], lpfa_total, 20)}  {pct_lpfa:5.1f}%')
    log(f'    {"High Voltage":<20}: {nn_result["hv"]:3d} / {hv_total:2d}  '
        f'  {bar(nn_result["hv"], hv_total, 20)}  {pct_hv:5.1f}%')

    # ─── Распределение всех проводов с отметкой NN ────────────────────────────

    log()
    log(f'  Распределение score по всем {len(df)} проводам:')
    n_bins   = 10
    bin_edges = np.linspace(real_score.min(), real_score.max(), n_bins + 1)

    for i in range(n_bins):
        lo, hi   = bin_edges[i], bin_edges[i + 1]
        mask     = (real_score >= lo) & (real_score < hi + 0.001)
        cnt      = mask.sum()
        bbar_s   = bar(cnt, len(df), 20)
        nn_here  = ' ← NN' if mask[nn_idx] else ''
        opt_here = ' ← OPT' if mask[optimum['idx']] else ''
        log(f'    [{lo:5.0f}–{hi:5.0f}]: {cnt:4d}  {bbar_s}{nn_here}{opt_here}')

    # ─── Топ-10 проводов по реальному score ───────────────────────────────────

    log()
    log(f'  Топ-10 проводов по реальному score:')
    log(f'  {"#":>3} {"Строка":>6} {"Score":>6} {"LP":>4} {"LPFA":>5} '
        f'{"HV":>4} {"GCSP":>7}  {"Gage":^14}  {"NN?":^5}')
    log('  ' + '─' * W)

    top10_idx = np.argsort(real_score)[::-1][:10]
    for rank, idx in enumerate(top10_idx, 1):
        row  = df.iloc[idx]
        mark = '← NN' if idx == nn_idx else ''
        log(f'  {rank:>3} {int(row["excel_row"]):>6} {int(real_score[idx]):>6} '
            f'{int(real_lp[idx]):>4} {int(real_lpfa[idx]):>5} '
            f'{int(real_hv[idx]):>4} {row["gcsp"]:>7.2f}  '
            f'[{row["min_gage"]:5.1f}–{row["max_gage"]:5.1f}]  {mark}')

    # ─── Вердикт ──────────────────────────────────────────────────────────────

    log()
    log('  ' + '═' * W)
    log(f'  {"📋  ВЕРДИКТ":^{W}}')
    log('  ' + '═' * W)
    log()

    # Оцениваем качество NN
    if pct_optimum >= 99.0:
        verdict = '🟢 ОТЛИЧНО  — NN нашла абсолютный оптимум'
    elif pct_optimum >= 95.0:
        verdict = '🟢 ХОРОШО   — NN очень близко к оптимуму'
    elif pct_optimum >= 90.0:
        verdict = '🟡 ПРИЕМЛЕМО — NN в пределах 10% от оптимума'
    elif pct_optimum >= 80.0:
        verdict = '🟠 СЛАБО    — NN упустила значимый прирост'
    else:
        verdict = '🔴 ПЛОХО    — NN сильно отклонилась от оптимума'

    log(f'  {verdict}')
    log()
    log(f'  Score NN-провода:      {nn_result["score"]:3d}  из {total_ops}'
        f'  ({pct_optimum:.1f}% от оптимума)')
    log(f'  Алгоритмический опт.:  {opt_score:3d}  из {total_ops}'
        f'  (строка {optimum["excel_row"]})')
    log(f'  Место NN в рейтинге:   #{rank_by_real} из {len(df)}')
    log(f'  Потеря операций:       {score_gap} операций  '
        f'({score_gap/total_ops*100:.1f}% от всех)')
    log()

    # Сравнение с наивным baseline (A6: просто самый быстрый провод)
    baseline_score = algo_results['A6_MIN_GCSP']['score']
    gain_vs_baseline = nn_result['score'] - baseline_score
    log(f'  NN vs baseline (быстрейший провод):')
    log(f'    Baseline score: {baseline_score}  '
        f'(строка {algo_results["A6_MIN_GCSP"]["excel_row"]})')
    log(f'    NN score:       {nn_result["score"]}')
    log(f'    Прирост NN:    +{gain_vs_baseline} операций  '
        f'({gain_vs_baseline/total_ops*100:.1f}% от всех)')


# ─── CSV результаты ───────────────────────────────────────────────────────────

rows_csv = []
for i, (_, row) in enumerate(df.iterrows()):
    is_nn  = (i == nn_idx) if nn_result else False
    is_opt = (i == optimum['idx'])
    rows_csv.append({
        'excel_row':  int(row['excel_row']),
        'machine_group': row['machine_group'],
        'min_gage':   row['min_gage'],
        'max_gage':   row['max_gage'],
        'min_len':    row['min_len'],
        'max_len':    row['max_len'],
        'gcsp':       row['gcsp'],
        'score_real': int(real_score[i]),
        'score_lp':   int(real_lp[i]),
        'score_lpfa': int(real_lpfa[i]),
        'score_hv':   int(real_hv[i]),
        'score_nn':   round(float(nn_preds[i]), 4) if nn_result else None,
        'is_nn_pick': int(is_nn),
        'is_optimum': int(is_opt),
        'rank_real':  rank_of(i, real_score),
        'pct_optimum': round(real_score[i] / opt_score * 100, 2),
    })

df_out = pd.DataFrame(rows_csv)
csv_path = os.path.join(OUTPUT_DIR, 'evaluation.csv')
df_out.to_csv(csv_path, index=False, encoding='utf-8-sig')
log()
log(f'  📁  Детальные данные → {csv_path}')

session_elapsed = (datetime.datetime.now() - session_start).total_seconds()
log(f'  ⏱  Время: {session_elapsed:.1f} сек')
log('  ' + '═' * W)

_log_file.close()