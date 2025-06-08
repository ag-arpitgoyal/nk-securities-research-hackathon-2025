import pandas as pd
import numpy as np
import re
from scipy.interpolate import CubicSpline
from tqdm import tqdm
import csv

# Load masked and original data
df_masked = pd.read_csv("new_train_unified_masked_filtered.csv")
df_full = pd.read_csv("new_train_unified_filtered.csv")

# Identify IV and X columns
iv_cols = [col for col in df_masked.columns if re.match(r'iv_\d+', col)]
x_cols = [col for col in df_masked.columns if col.startswith('X')]

# Strike map from column names
strike_map = {col: int(re.search(r'\d+', col).group()) for col in iv_cols}

# Output CSV file
output_file = "new_train_residuals_masked_filtered.csv"

# Prepare header for the output file
output_columns = ['timestamp', 'underlying'] + iv_cols + x_cols

# Open the output CSV in write mode
with open(output_file, mode='w', newline='') as f_out:
    writer = csv.DictWriter(f_out, fieldnames=output_columns)
    writer.writeheader()

    for idx in tqdm(range(len(df_masked))):
        row_masked = df_masked.iloc[idx]
        row_full = df_full.iloc[idx]

        residual_row = {
            'timestamp': row_masked['timestamp'],
            'underlying': row_masked['underlying']
        }

        for col in x_cols:
            residual_row[col] = row_masked[col]

        underlying_price = row_masked['underlying']
        if pd.isna(underlying_price) or underlying_price == 0:
            for col in iv_cols:
                residual_row[col] = np.nan
            writer.writerow(residual_row)
            continue

        # Build IV data for spline
        iv_data = {}
        for col in iv_cols:
            strike = strike_map[col]
            iv = row_masked[col]
            if pd.notna(iv):
                moneyness = strike / underlying_price
                iv_data[moneyness] = iv

        if len(iv_data) < 3:
            for col in iv_cols:
                residual_row[col] = np.nan
            writer.writerow(residual_row)
            continue

        # Prepare moneyness and IV arrays
        strikes, ivs = zip(*sorted(iv_data.items()))
        strikes = np.array(strikes)
        ivs = np.array(ivs)
        moneyness_underlying = 1.0

        left_mask = strikes < moneyness_underlying
        right_mask = strikes > moneyness_underlying
        num_left = np.sum(left_mask)
        num_right = np.sum(right_mask)

        if num_left < 2 and num_right >= 2:
            mirror_strikes = 2 * moneyness_underlying - strikes[right_mask]
            mirror_ivs = ivs[right_mask]
            strikes = np.concatenate([strikes, mirror_strikes])
            ivs = np.concatenate([ivs, mirror_ivs])
        elif num_right < 2 and num_left >= 2:
            mirror_strikes = 2 * moneyness_underlying - strikes[left_mask]
            mirror_ivs = ivs[left_mask]
            strikes = np.concatenate([mirror_strikes, strikes])
            ivs = np.concatenate([mirror_ivs, ivs])
        elif num_left < 2 and num_right < 2:
            for col in iv_cols:
                residual_row[col] = np.nan
            writer.writerow(residual_row)
            continue

        sorted_idx = np.argsort(strikes)
        strikes = strikes[sorted_idx]
        ivs = ivs[sorted_idx]

        idx_underlying = np.abs(strikes - moneyness_underlying).argmin()
        strike_underlying = strikes[idx_underlying]
        iv_underlying = ivs[idx_underlying]

        for i in range(1, len(strikes)):
            if strikes[i] != strikes[0]:
                slope_left_linear = 1.5 * (ivs[i] - ivs[0]) / (strikes[i] - strikes[0])
                slope_left = (
                    1.5 * (iv_underlying - ivs[0]) / (strike_underlying - strikes[0])
                    if strike_underlying != strikes[0]
                    else slope_left_linear
                )
                break

        for i in range(1, len(strikes)):
            if strikes[-i - 1] != strikes[-1]:
                slope_right_linear = 1.5 * (ivs[-1] - ivs[-i - 1]) / (strikes[-1] - strikes[-i - 1])
                slope_right = (
                    1.5 * (ivs[-1] - iv_underlying) / (strikes[-1] - strike_underlying)
                    if strike_underlying != strikes[-1]
                    else slope_right_linear
                )
                break

        spline = CubicSpline(strikes, ivs, bc_type=((1, slope_left), (1, slope_right)), extrapolate=False)

        for col in iv_cols:
            strike = strike_map[col]
            moneyness = strike / underlying_price
            was_masked = pd.isna(row_masked[col])
            if not was_masked:
                residual_row[col] = np.nan
                continue

            if moneyness < strikes[0]:
                pred_iv = ivs[0] + slope_left_linear * (moneyness - strikes[0])
            elif moneyness > strikes[-1]:
                pred_iv = ivs[-1] + slope_right_linear * (moneyness - strikes[-1])
            else:
                pred_iv = spline(moneyness)

            pred_iv = np.clip(pred_iv, 0.12, 0.55)
            true_iv = row_full[col]
            residual_row[col] = true_iv - pred_iv

        writer.writerow(residual_row)
