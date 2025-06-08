import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from scipy.interpolate import CubicSpline
from tqdm import tqdm

# Load data
df = pd.read_csv("final_test_unified.csv")

# Identify IV columns and strike values
iv_cols = sorted([col for col in df.columns if re.match(r'iv_\d+', col)],
                 key=lambda x: int(x.split('_')[1]))
strikes = [int(col.split('_')[1]) for col in iv_cols]
strike_map = {col: int(re.search(r'\d+', col).group()) for col in iv_cols}

# Flagging logic
def check_row(row):
    non_nan_strikes = [strike for strike, col in zip(strikes, iv_cols) if pd.notna(row[col])]
    for i in range(1, len(non_nan_strikes)):
        if non_nan_strikes[i] - non_nan_strikes[i - 1] >= 1400:
            return True
    iv_values = row[iv_cols].values
    left_missing = sum(1 for val in iv_values[:9] if pd.isna(val))
    right_missing = sum(1 for val in iv_values[-9:] if pd.isna(val))
    if left_missing >= 9 or right_missing >= 9:
        return True
    if np.count_nonzero(pd.notna(iv_values)) < 11:
        return True
    return False

df['flag'] = df.apply(check_row, axis=1)

# Prepare output file and write header
output_file = "v4_manual.csv"
df.iloc[0:0].drop(columns=['flag']).to_csv(output_file, index=False)

# Spline fitting helper
def fit_spline_with_manual_extrapolation(strike_arr, iv_arr):
    if len(strike_arr) >= 3:
        slope_left = 1.5 * (iv_arr[1] - iv_arr[0]) / (strike_arr[1] - strike_arr[0])
        slope_right = 1.5 * (iv_arr[-1] - iv_arr[-2]) / (strike_arr[-1] - strike_arr[-2])
        spline = CubicSpline(strike_arr, iv_arr, bc_type=((1, slope_left), (1, slope_right)), extrapolate=False)
        return spline, slope_left, slope_right, strike_arr[0], strike_arr[-1], iv_arr[0], iv_arr[-1]
    return None, None, None, None, None, None, None

# Interactive correction function
def handle_flagged_row(row_idx, row):
    added_points = []
    done = [False]

    underlying = row['underlying']
    if pd.isna(underlying) or underlying <= 0:
        print(f"Row {row_idx} skipped (invalid underlying)")
        return False

    known_points = [(strike, row[f'iv_{strike}']) for strike in strikes if pd.notna(row[f'iv_{strike}'])]
    if len(known_points) < 2:
        print(f"Row {row_idx} skipped (not enough known points)")
        return False

    known_strikes = np.array([s for s, iv in known_points])
    known_ivs = np.array([iv for s, iv in known_points])

    # Mirroring logic
    left_mask = known_strikes < underlying
    right_mask = known_strikes > underlying
    if np.sum(left_mask) < 2 and np.sum(right_mask) >= 2:
        mirror_strikes = 2 * underlying - known_strikes[right_mask]
        mirror_ivs = known_ivs[right_mask]
        known_strikes = np.concatenate([known_strikes, mirror_strikes])
        known_ivs = np.concatenate([known_ivs, mirror_ivs])
    elif np.sum(right_mask) < 2 and np.sum(left_mask) >= 2:
        mirror_strikes = 2 * underlying - known_strikes[left_mask]
        mirror_ivs = known_ivs[left_mask]
        known_strikes = np.concatenate([mirror_strikes, known_strikes])
        known_ivs = np.concatenate([mirror_ivs, known_ivs])

    sort_idx = np.argsort(known_strikes)
    all_strikes = list(known_strikes[sort_idx])
    all_ivs = list(known_ivs[sort_idx])

    fig, ax = plt.subplots(figsize=(10, 5))
    try:
        mng = plt.get_current_fig_manager()
        if hasattr(mng, 'window'):
            screen_width = mng.window.winfo_screenwidth()
            screen_height = mng.window.winfo_screenheight()
            fig_width = 1000
            fig_height = 600
            x = int((screen_width - fig_width) / 2)
            y = int((screen_height - fig_height) / 2)
            mng.window.wm_geometry(f"{fig_width}x{fig_height}+{x}+{y}")
    except Exception as e:
        print(f"Could not center window: {e}")

    ax.set_xlim(22800, 26700)
    ax.set_ylim(0.12, 0.55)
    ax.set_title(f"Row {row_idx} - Click to Add Points")
    ax.set_xlabel("Strike")
    ax.set_ylabel("IV")
    ax.grid(True)
    ax.scatter(all_strikes, all_ivs, color='blue', label='Known IVs')

    spline_line, = ax.plot([], [], '--', color='green', label='Spline')
    added_scatter = ax.scatter([], [], color='red', label='Added')
    ax.legend()

    x_grid = np.linspace(23000, 26500, 300)

    def update_plot():
        nonlocal all_strikes, all_ivs
        spline, slope_l, slope_r, s0, s1, iv0, iv1 = fit_spline_with_manual_extrapolation(np.array(all_strikes), np.array(all_ivs))
        if spline is not None:
            y_spline = []
            for x in x_grid:
                if x < s0:
                    y = iv0 + slope_l * (x - s0)
                elif x > s1:
                    y = iv1 + slope_r * (x - s1)
                else:
                    y = spline(x)
                y_spline.append(np.clip(y, 0.12, 0.55))
            spline_line.set_data(x_grid, y_spline)
        else:
            spline_line.set_data([], [])
        added_scatter.set_offsets(np.array(added_points) if added_points else np.empty((0, 2)))
        fig.canvas.draw_idle()

    def onclick(event):
        nonlocal all_strikes, all_ivs
        if event.inaxes != ax:
            return
        x, y = event.xdata, event.ydata
        added_points.append((x, y))
        all_strikes.append(x)
        all_ivs.append(y)
        sort_idx = np.argsort(all_strikes)
        all_strikes = list(np.array(all_strikes)[sort_idx])
        all_ivs = list(np.array(all_ivs)[sort_idx])
        update_plot()

    def on_done(event):
        done[0] = True
        plt.close(fig)

    fig.canvas.mpl_connect('button_press_event', onclick)
    ax_done = plt.axes([0.85, 0.025, 0.1, 0.04])
    button = Button(ax_done, 'Done')
    button.on_clicked(on_done)

    update_plot()
    plt.show(block=True)

    if not added_points:
        print("No points added. Skipping row.")
        return False

    spline, slope_l, slope_r, s0, s1, iv0, iv1 = fit_spline_with_manual_extrapolation(np.array(all_strikes), np.array(all_ivs))
    for col in iv_cols:
        strike = strike_map[col]
        if pd.isna(row[col]):
            if strike < s0:
                val = iv0 + slope_l * (strike - s0)
            elif strike > s1:
                val = iv1 + slope_r * (strike - s1)
            else:
                val = spline(strike)
            df.at[row_idx, col] = np.clip(val, 0.12, 0.55)

    print(f"✔ Row {row_idx} updated with {len(added_points)} points.")
    return True

# Process only flagged rows and save in real-time
for idx in tqdm(df.index[df['flag']]):
    if handle_flagged_row(idx, df.loc[idx]):
        df.iloc[[idx]].drop(columns=['flag']).to_csv(output_file, mode='a', index=False, header=False)

print("✅ All flagged rows processed and saved to v4_manual.csv")
