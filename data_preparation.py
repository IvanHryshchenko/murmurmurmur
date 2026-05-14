import pandas as pd
import numpy as np


def load_and_prepare_k1(file_path='K1_old.xlsx'):

    xl = pd.ExcelFile(file_path)

    df_original = xl.parse('TOTAL values GCSD',header=None)
    data = []

    for i in range(len(df_original)):

        row = df_original.iloc[i].values

        numeric_values = []
        part_name = None

        for cell in row:

            if pd.isna(cell):
                continue

            try:
                val = float(cell)

                if val > 0.001:
                    numeric_values.append(val)

            except ValueError:

                s = str(cell).strip()

                if (
                    2 < len(s) < 100
                    and not s.startswith('TOTAL')
                    and 'Wires number' not in s
                ):
                    part_name = s

        if len(numeric_values) >= 2 and part_name:

            gcsd = numeric_values[0]
            adj = numeric_values[1]

            plant = (
                numeric_values[2]
                if len(numeric_values) > 2
                else None
            )

            data.append({
                'row_index': i,
                'PART NUMBER': part_name,
                'GCSD': gcsd,
                'ADJ.': adj,
                'PLANT STANDARD': plant
            })

    df_clean = pd.DataFrame(data)

    print(f'✅ Извлечено строк: {len(df_clean)}')

    return df_clean, df_original


def create_features(df):
    df = df.copy()
    df['GCSD_log'] = np.log1p(df['GCSD'])
    df['ADJ_log'] = np.log1p(df['ADJ.'])
    df['GCSD_x_ADJ'] = df['GCSD'] * df['ADJ.']
    df['GCSD_sqrt'] = np.sqrt(df['GCSD'])
    df['ADJ_sqrt'] = np.sqrt(df['ADJ.'])
    df['PART_LEN'] = (
        df['PART NUMBER']
        .astype(str)
        .str.len()
    )

    df['PART_DIGITS'] = (
        df['PART NUMBER']
        .astype(str)
        .str.count(r'\d')
    )

    df['PART_HAS_LETTER'] = (
        df['PART NUMBER']
        .astype(str)
        .str.contains(r'[A-Za-z]')
        .astype(int)
    )

    df['PLANT STANDARD'] = df['PLANT STANDARD'].fillna(
        df['GCSD'] * df['ADJ.']
    )

    return df