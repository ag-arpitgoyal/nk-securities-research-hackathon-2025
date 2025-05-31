import pandas as pd
import numpy as np
from numpy.polynomial.polynomial import Polynomial
from joblib import Parallel, delayed
from tqdm import tqdm

# Load test data
test = pd.read_csv("test_data.csv")

# Define IV columns
call_iv_cols = [col for col in test.columns if col.startswith("call_iv_")]
put_iv_cols = [col for col in test.columns if col.startswith("put_iv_")]

# Get strike prices
def extract_strike(col): return int(col.split("_")[-1])
call_strikes = np.array([extract_strike(c) for c in call_iv_cols])
put_strikes = np.array([extract_strike(p) for p in put_iv_cols])

# Fit and fill missing values using polynomial smile
def fill_row_iv_smile(row_vals, strikes):
    mask = ~np.isnan(row_vals)
    if mask.sum() < 3:
        return row_vals  # not enough to fit
    coefs = Polynomial.fit(strikes[mask], row_vals[mask], deg=2).convert().coef
    fitted_vals = Polynomial(coefs)(strikes)
    row_vals[~mask] = fitted_vals[~mask]
    return row_vals

# Wrap to process one row of test data
def process_row(row):
    call_vals = row[call_iv_cols].values.astype(float)
    put_vals = row[put_iv_cols].values.astype(float)
    filled_call = fill_row_iv_smile(call_vals, call_strikes)
    filled_put = fill_row_iv_smile(put_vals, put_strikes)
    return np.concatenate([filled_call, filled_put])

# Run in parallel (with progress bar)
results = Parallel(n_jobs=-1, backend='loky')(
    delayed(process_row)(row) for _, row in tqdm(test.iterrows(), total=len(test))
)

# Assign results back
iv_data = np.array(results)

# Assign results to a DataFrame
iv_df = pd.DataFrame(iv_data, columns=call_iv_cols + put_iv_cols)

# Step 1: Fill row-wise NaNs with row mean
iv_df = iv_df.apply(lambda row: row.fillna(row.mean()), axis=1)

# Step 2: Fill remaining NaNs (if entire row section is NaN) with column mean
iv_df.fillna(iv_df.mean(), inplace=True)

# Assign back to test
test[call_iv_cols + put_iv_cols] = iv_df

# Save final submission
submission = test[["timestamp"] + call_iv_cols + put_iv_cols]
submission.to_csv("v1_2.csv", index=False)

