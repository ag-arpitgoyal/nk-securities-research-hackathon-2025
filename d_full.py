import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from ipywidgets import Button, Output, VBox
from IPython.display import display

# Load data
df = pd.read_csv("final_test_unified.csv")

# Identify IV columns and corresponding strikes
iv_cols = [col for col in df.columns if re.match(r'iv_\d+', col)]
iv_cols.sort(key=lambda x: int(x.split('_')[1]))
strikes = [int(col.split('_')[1]) for col in iv_cols]

# Flagging function
def check_row(row):
    non_nan_strikes = [strike for strike, col in zip(strikes, iv_cols) if pd.notna(row[col])]
    for i in range(1, len(non_nan_strikes)):
        if non_nan_strikes[i] - non_nan_strikes[i - 1] >= 1100:
            return True

    iv_values = row[iv_cols].values
    left_missing = 0
    for val in iv_values:
        if pd.isna(val):
            left_missing += 1
        else:
            break
    right_missing = 0
    for val in reversed(iv_values):
        if pd.isna(val):
            right_missing += 1
        else:
            break
    if left_missing >= 9 or right_missing >= 9:
        return True

    if np.count_nonzero(pd.notna(iv_values)) < 11:
        return True

    return False

# Apply flag
df['flag'] = df.apply(check_row, axis=1)

# Prepare interactive output
output = Output()

# Get index of first flagged row
flagged_indices = df[df['flag']].index.tolist()
if not flagged_indices:
    raise ValueError("No flagged rows to edit.")

row_idx = flagged_indices[0]
row = df.loc[row_idx]

# Extract known IVs
known_points = [(strike, row[f'iv_{strike}']) for strike in strikes if pd.notna(row[f'iv_{strike}'])]
known_strikes = np.array([s for s, iv in known_points])
known_ivs = np.array([iv for s, iv in known_points])

# Container for user-added points
added_points = []

# Plot and click handler
fig, ax = plt.subplots(figsize=(10, 5))
scatter = ax.scatter(known_strikes, known_ivs, label='Known IVs', color='blue')
line = None

def onclick(event):
    if event.inaxes != ax:
        return
    added_points.append((event.xdata, event.ydata))
    ax.scatter(event.xdata, event.ydata, color='red', label='Added point' if len(added_points) == 1 else "")
    fig.canvas.draw()

def on_done_clicked(b):
    all_strikes = np.concatenate([known_strikes, [p[0] for p in added_points]])
    all_ivs = np.concatenate([known_ivs, [p[1] for p in added_points]])
    sorted_idx = np.argsort(all_strikes)
    all_strikes = all_strikes[sorted_idx]
    all_ivs = all_ivs[sorted_idx]

    # Estimate boundary slopes
    slope_left = 1.5 * (all_ivs[1] - all_ivs[0]) / (all_strikes[1] - all_strikes[0])
    slope_right = 1.5 * (all_ivs[-1] - all_ivs[-2]) / (all_strikes[-1] - all_strikes[-2])

    # Fit spline
    spline = CubicSpline(all_strikes, all_ivs, bc_type=((1, slope_left), (1, slope_right)), extrapolate=True)

    # Fill the row
    filled_values = []
    for strike in strikes:
        iv = row[f'iv_{strike}']
        if pd.notna(iv):
            filled_values.append(iv)
        else:
            val = spline(strike)
            filled_values.append(np.clip(val, 0.12, 0.55))

    # Output filled row
    filled_row = pd.Series(filled_values, index=iv_cols)
    output.clear_output()
    with output:
        print(f"Filled row for index {row_idx}:")
        display(filled_row)

# Attach button
done_button = Button(description="Done - Refit Spline")
done_button.on_click(on_done_clicked)

# Set up plot
fig.canvas.mpl_connect('button_press_event', onclick)
ax.set_title(f"Row {row_idx} - Click to Add Points")
ax.set_xlabel("Strike")
ax.set_ylabel("IV")
ax.legend()
plt.show()

# Display everything
display(VBox([done_button, output]))
