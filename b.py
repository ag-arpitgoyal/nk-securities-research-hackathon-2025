import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
from joblib import Parallel, delayed
from tqdm import tqdm

# Load test data
test = pd.read_csv("test_data.csv")

# Define IV columns
call_iv_cols = [col for col in test.columns if col.startswith("call_iv_")]
put_iv_cols = [col for col in test.columns if col.startswith("put_iv_")]

# Extract strike prices
def extract_strike(col): return int(col.split("_")[-1])
call_strikes = np.array([extract_strike(c) for c in call_iv_cols])
put_strikes = np.array([extract_strike(p) for p in put_iv_cols])

# Fill missing IVs using cubic spline (with linear fallback if <4 points)
def fill_iv_spline(row_vals, strikes):
    mask = ~np.isnan(row_vals)
    if mask.sum() < 2:
        return row_vals  # can't interpolate
    x = strikes[mask]
    y = row_vals[mask]

    try:
        if len(x) >= 4:
            spline = CubicSpline(x, y, extrapolate=True)
        else:
            # Fallback to linear interpolation
            spline = lambda z: np.interp(z, x, y)

        row_vals[~mask] = spline(strikes[~mask])
    except:
        pass  # if spline fails for some reason, skip
    return row_vals

# Process a single row
def process_row(row):
    call_vals = row[call_iv_cols].values.astype(float)
    put_vals = row[put_iv_cols].values.astype(float)
    filled_call = fill_iv_spline(call_vals, call_strikes)
    filled_put = fill_iv_spline(put_vals, put_strikes)
    return np.concatenate([filled_call, filled_put])

# Run with parallel processing and progress bar
results = Parallel(n_jobs=-1, backend='loky')(
    delayed(process_row)(row) for _, row in tqdm(test.iterrows(), total=len(test))
)

# Update test DataFrame
iv_data = np.array(results)
iv_df = pd.DataFrame(iv_data, columns=call_iv_cols + put_iv_cols)

# Step 1: Fill row-wise NaNs with row mean
iv_df = iv_df.apply(lambda row: row.fillna(row.mean()), axis=1)

# Step 2: Fill any remaining NaNs (e.g., entire row NaN) with column mean
iv_df.fillna(iv_df.mean(), inplace=True)

# Assign back to test DataFrame
test[call_iv_cols + put_iv_cols] = iv_df

# Save submission
submission = test[["timestamp"] + call_iv_cols + put_iv_cols]
submission.to_csv("v2.csv", index=False)
