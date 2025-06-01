import pandas as pd
import numpy as np
import re
from scipy.interpolate import CubicSpline

# Load data
df = pd.read_csv('test_data.csv')

# Identify IV columns and strike values
iv_cols = [col for col in df.columns if re.match(r'(call_iv_|put_iv_)\d+', col)]
strike_map = {col: int(re.search(r'\d+', col).group()) for col in iv_cols}

# Extract unique strikes
all_strikes = sorted(set(strike_map.values()))

# Prepare output DataFrame
output_df = df[['timestamp']].copy()

# Function to fill missing IVs for a single row
def fill_missing_ivs(row):
    iv_data = {}
    for col in iv_cols:
        strike = strike_map[col]
        iv = row[col]
        if pd.notna(iv):
            iv_data[strike] = iv

    if len(iv_data) < 3:
        return [row[col] for col in iv_cols]  # Not enough points for spline

    # Prepare data
    strikes, ivs = zip(*sorted(iv_data.items()))
    strikes = np.array(strikes)
    ivs = np.array(ivs)
    underlying_price = row['underlying']

    # Mirroring logic
    left_mask = strikes < underlying_price
    right_mask = strikes > underlying_price
    num_left = np.sum(left_mask)
    num_right = np.sum(right_mask)

    if num_left < 2 and num_right >= 2:
        mirror_strikes = 2 * underlying_price - strikes[right_mask]
        mirror_ivs = ivs[right_mask]
        strikes = np.concatenate([strikes, mirror_strikes])
        ivs = np.concatenate([ivs, mirror_ivs])
    elif num_right < 2 and num_left >= 2:
        mirror_strikes = 2 * underlying_price - strikes[left_mask]
        mirror_ivs = ivs[left_mask]
        strikes = np.concatenate([mirror_strikes, strikes])
        ivs = np.concatenate([mirror_ivs, ivs])
    elif num_left < 2 and num_right < 2:
        return [row[col] for col in iv_cols]  # Still not enough valid data

    sorted_idx = np.argsort(strikes)
    strikes = strikes[sorted_idx]
    ivs = ivs[sorted_idx]

    # Get slope estimates
    idx_underlying = np.abs(strikes - underlying_price).argmin()
    strike_underlying = strikes[idx_underlying]
    iv_underlying = ivs[idx_underlying]

    for i in range(1, len(strikes)):
        if strikes[i] != strikes[0]:
            slope_left_linear = (ivs[i] - ivs[0]) / (strikes[i] - strikes[0])
            slope_left = (
                1.5 * (iv_underlying - ivs[0]) / (strike_underlying - strikes[0])
                if strike_underlying != strikes[0]
                else slope_left_linear
            )
            break

    for i in range(1, len(strikes)):
        if strikes[-i - 1] != strikes[-1]:
            slope_right_linear = (ivs[-1] - ivs[-i - 1]) / (strikes[-1] - strikes[-i - 1])
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
        iv = row[col]
        if pd.notna(iv):
            filled_ivs.append(iv)
        elif strike < strikes[0]:
            filled_ivs.append(ivs[0] + slope_left_linear * (strike - strikes[0]))
        elif strike > strikes[-1]:
            filled_ivs.append(ivs[-1] + slope_right_linear * (strike - strikes[-1]))
        else:
            filled_ivs.append(spline(strike))

    return filled_ivs

# Apply to all rows
filled_ivs_array = df.apply(fill_missing_ivs, axis=1, result_type='expand')
filled_ivs_array.columns = iv_cols

# Save final DataFrame
output_df = pd.concat([output_df, filled_ivs_array], axis=1)
output_df.to_csv('v3.csv', index=False)

print("Missing IVs filled and saved to v3.csv")
