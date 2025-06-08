import pandas as pd
import numpy as np
import re
from scipy.interpolate import CubicSpline
from tqdm import tqdm

# Load data
df = pd.read_csv('final_test_unified.csv')

# Identify IV columns and strike values
iv_cols = [col for col in df.columns if re.match(r'(iv_)\d+', col)]
strike_map = {col: int(re.search(r'\d+', col).group()) for col in iv_cols}

# Output initialization
output_df = df[['timestamp']].copy()

# Function to fill missing IVs using cubic spline in strike space, mirroring around underlying
def fill_missing_ivs(row):
    underlying = row['underlying']
    if pd.isna(underlying) or underlying == 0:
        return [row[col] for col in iv_cols]

    known_strikes = []
    known_ivs = []

    for col in iv_cols:
        if pd.notna(row[col]):
            known_strikes.append(strike_map[col])
            known_ivs.append(row[col])

    if len(known_strikes) < 3:
        return [row[col] for col in iv_cols]

    strikes = np.array(known_strikes)
    ivs = np.array(known_ivs)

    # Sort by strike
    sort_idx = np.argsort(strikes)
    strikes = strikes[sort_idx]
    ivs = ivs[sort_idx]

    # Mirroring around underlying
    left_mask = strikes < underlying
    right_mask = strikes > underlying
    num_left = np.sum(left_mask)
    num_right = np.sum(right_mask)

    if num_left < 2 and num_right >= 2:
        mirror_strikes = 2 * underlying - strikes[right_mask]
        mirror_ivs = ivs[right_mask]
        strikes = np.concatenate([strikes, mirror_strikes])
        ivs = np.concatenate([ivs, mirror_ivs])
    elif num_right < 2 and num_left >= 2:
        mirror_strikes = 2 * underlying - strikes[left_mask]
        mirror_ivs = ivs[left_mask]
        strikes = np.concatenate([mirror_strikes, strikes])
        ivs = np.concatenate([mirror_ivs, ivs])
    elif num_left < 2 and num_right < 2:
        return [row[col] for col in iv_cols]

    # Sort again
    sort_idx = np.argsort(strikes)
    strikes = strikes[sort_idx]
    ivs = ivs[sort_idx]

    # Estimate boundary slopes
    slope_left = 1.5 * (ivs[1] - ivs[0]) / (strikes[1] - strikes[0])
    slope_right = 1.5 * (ivs[-1] - ivs[-2]) / (strikes[-1] - strikes[-2])

    # Fit spline
    spline = CubicSpline(strikes, ivs, bc_type=((1, slope_left), (1, slope_right)), extrapolate=False)

    # Fill all missing IVs
    filled_ivs = []
    for col in iv_cols:
        strike = strike_map[col]
        iv = row[col]
        if pd.notna(iv):
            filled_ivs.append(iv)
        elif strike < strikes[0]:
            val = ivs[0] + slope_left * (strike - strikes[0])
            filled_ivs.append(np.clip(val, 0.12, 0.55))
        elif strike > strikes[-1]:
            val = ivs[-1] + slope_right * (strike - strikes[-1])
            filled_ivs.append(np.clip(val, 0.12, 0.55))
        else:
            val = spline(strike)
            filled_ivs.append(np.clip(val, 0.12, 0.55))

    return filled_ivs

# Apply with progress bar
tqdm.pandas()
filled = df.progress_apply(fill_missing_ivs, axis=1, result_type='expand')
filled.columns = iv_cols

# Combine and save
output_df = pd.concat([output_df, filled], axis=1)
output_df.to_csv('v4_strike_spline_underlying_mirror.csv', index=False)

print("IVs filled using strike-based spline with mirroring around underlying. Saved to v4_strike_spline_underlying_mirror.csv.")
