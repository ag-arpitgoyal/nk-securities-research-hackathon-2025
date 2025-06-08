import pandas as pd
import numpy as np
import re
from scipy.interpolate import CubicSpline
from tqdm import tqdm

# Load data
df = pd.read_csv('final_test_unified_half.csv')

# Identify IV columns and strike values
iv_cols = [col for col in df.columns if re.match(r'(iv_)\d+', col)]
strike_map = {col: int(re.search(r'\d+', col).group()) for col in iv_cols}

# Prepare output DataFrame
output_df = df[['timestamp']].copy()

# Function to fill missing IVs for a single row
def fill_missing_ivs(row):
    iv_data = {}
    underlying_price = row['underlying']

    if pd.isna(underlying_price) or underlying_price == 0:
        return [row[col] for col in iv_cols]  # Skip if underlying is missing or zero

    for col in iv_cols:
        strike = strike_map[col]
        iv = row[col]
        if pd.notna(iv):
            moneyness = strike / underlying_price
            iv_data[moneyness] = iv

    if len(iv_data) < 3:
        return [row[col] for col in iv_cols]  # Not enough points for spline

    # Prepare data
    strikes, ivs = zip(*sorted(iv_data.items()))
    strikes = np.array(strikes)
    ivs = np.array(ivs)
    moneyness_underlying = 1.0

    # Mirroring logic (in moneyness space)
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
        return [row[col] for col in iv_cols]  # Still not enough valid data

    sorted_idx = np.argsort(strikes)
    strikes = strikes[sorted_idx]
    ivs = ivs[sorted_idx]

    # Get slope estimates
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

    # Fit spline
    spline = CubicSpline(strikes, ivs, bc_type=((1, slope_left), (1, slope_right)), extrapolate=False)

    # Fill missing values
    filled_ivs = []
    for col in iv_cols:
        strike = strike_map[col]
        moneyness = strike / underlying_price
        iv = row[col]
        if pd.notna(iv):
            filled_ivs.append(iv)
        elif moneyness < strikes[0]:
            val = ivs[0] + slope_left_linear * (moneyness - strikes[0])
            filled_ivs.append(np.clip(val, 0.12, 0.55))
        elif moneyness > strikes[-1]:
            val = ivs[-1] + slope_right_linear * (moneyness - strikes[-1])
            filled_ivs.append(np.clip(val, 0.12, 0.55))
        else:
            val = spline(moneyness)
            filled_ivs.append(np.clip(val, 0.12, 0.55))

    return filled_ivs

# Enable progress bar
tqdm.pandas()

# Apply interpolation row-wise with progress bar
filled_ivs_array = df.progress_apply(fill_missing_ivs, axis=1, result_type='expand')
filled_ivs_array.columns = iv_cols

# Combine with timestamp and save
output_df = pd.concat([output_df, filled_ivs_array], axis=1)
output_df.to_csv('v4_manual_full.csv', index=False)

print("Missing IVs filled using normalized strikes and clipping, saved to v4_manual_full.csv")
