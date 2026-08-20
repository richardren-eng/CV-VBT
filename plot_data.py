import os
import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks

# --- STEP 0: FILE SELECTION DIALOG ---
root = tk.Tk()
root.withdraw()
root.attributes("-topmost", True)

csv_path = filedialog.askopenfilename(
    title="Select CV-VBT Trajectory CSV",
    initialdir="outputs",
    filetypes=[
        ("CSV Files", "*.csv"),
        ("All Files", "*.*")
    ]
)
root.destroy()

if not csv_path:
    print("No CSV file selected. Exiting.")
    exit()

file_label = os.path.splitext(os.path.basename(csv_path))[0]

# --- STEP 1: FILTERING & DATA INGESTION ---
def butter_lowpass_filter(data, cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

df = pd.read_csv(csv_path).interpolate(method='linear').dropna()

time = df["time_s"].to_numpy()
x_px = df["x_px"].to_numpy()
y_px = df["y_px"].to_numpy()

dt = np.diff(time)
fs = 1.0 / np.median(dt)
cutoff = 6.0  # Cutoff frequency (Hz)

print(f"Analyzing {file_label}: {len(df)} frames at {fs:.2f} Hz")

# --- STEP 2: KINEMATICS & DERIVATIVES ---
x_filtered = butter_lowpass_filter(x_px, cutoff, fs)
y_filtered = butter_lowpass_filter(y_px, cutoff, fs)

vx_filt = np.gradient(x_filtered, time)
vy_filt = -np.gradient(y_filtered, time)  # Upward motion positive

vx_raw = np.gradient(x_px, time)
vy_raw = -np.gradient(y_px, time)

# --- STEP 3: CONCENTRIC REP SEGMENTATION ---
peaks, _ = find_peaks(vy_filt, height=100, distance=int(fs * 1.5))

if len(peaks) == 0:
    raise RuntimeError("No concentric reps detected. Adjust peak height threshold or check coordinates.")

rep_metrics = []
concentric_windows = []

for i, peak_idx in enumerate(peaks, 1):
    left = peak_idx
    while left > 0 and vy_filt[left] > 15.0:
        left -= 1
    right = peak_idx
    while right < len(vy_filt) - 1 and vy_filt[right] > 15.0:
        right += 1
        
    concentric_windows.append((left, right))
    concentric_vy = vy_filt[left:right]
    mean_vy = np.mean(concentric_vy)
    peak_vy = vy_filt[peak_idx]
    rep_metrics.append({"rep": i, "mean_vy": mean_vy, "peak_vy": peak_vy})

rep_df = pd.DataFrame(rep_metrics)

# --- STEP 4: VELOCITY LOSS CALCULATION ---
rep1_mean = rep_df.loc[0, "mean_vy"]
rep_df["mean_pct_of_rep1"] = (rep_df["mean_vy"] / rep1_mean) * 100
rep_df["mean_loss_pct"] = ((rep1_mean - rep_df["mean_vy"]) / rep1_mean) * 100

# --- STEP 5: VISUALIZATION (OUTSIDE LEGENDS & EXTENDED MARGINS) ---
fig = plt.figure(figsize=(13.5, 10.5))

# Reserve the right 22% of the canvas for legends to prevent plot data overlap
gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1.2], hspace=0.35, left=0.08, right=0.76, top=0.93, bottom=0.07)

ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1], sharex=ax1)
ax3 = fig.add_subplot(gs[2])

# Panel 1: Horizontal Velocity
ax1.plot(time, vx_raw, color="lightcoral", linewidth=0.9, alpha=0.5, label="Raw $v_x$ (Unfiltered)")
ax1.plot(time, vx_filt, color="darkred", linewidth=1.8, label=f"Filtered $v_x$ ($f_c={cutoff}$ Hz)")
ax1.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
ax1.set_ylabel("Horizontal\nVelocity (px/s)", fontsize=10)
ax1.set_title(f"Barbell Kinematics & Velocity Fatigue Analysis — {file_label}", fontsize=12, fontweight="bold", loc="left")
ax1.grid(True, linestyle=":", alpha=0.6)
ax1.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0, framealpha=0.9, fontsize=9)

# Panel 2: Vertical Velocity with Concentric Highlights
ax2.plot(time, vy_raw, color="cornflowerblue", linewidth=0.9, alpha=0.5, label="Raw $v_y$ (Unfiltered)")
ax2.plot(time, vy_filt, color="navy", linewidth=1.8, label=f"Filtered $v_y$ ($f_c={cutoff}$ Hz)")
ax2.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)

for idx, (l, r) in enumerate(concentric_windows):
    ax2.axvspan(time[l], time[r], color="royalblue", alpha=0.15, 
                label="Concentric Phase" if idx == 0 else "")
    ax2.text(time[peaks[idx]], vy_filt[peaks[idx]] + 15, f"R{idx+1}", 
             ha="center", fontsize=9, fontweight="bold", color="navy")

ax2.set_xlabel("Time (s)", fontsize=10)
ax2.set_ylabel("Vertical\nVelocity (px/s)", fontsize=10)
ax2.grid(True, linestyle=":", alpha=0.6)
ax2.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0, framealpha=0.9, fontsize=9)

# Panel 3: Mean Concentric Velocity Loss (Bar Chart)
reps = [f"Rep {r}" for r in rep_df["rep"]]
bars = ax3.bar(reps, rep_df["mean_pct_of_rep1"], width=0.45, color="steelblue", edgecolor="black", linewidth=1.2)

ax3.axhline(100, color="darkgreen", linestyle="--", linewidth=1.2, label="Rep 1 Baseline (100%)")
ax3.axhline(80, color="goldenrod", linestyle="--", linewidth=1.2, label="20% Loss (~RPE 7.5–8.0)")
ax3.axhline(65, color="crimson", linestyle="--", linewidth=1.2, label="35% Loss (~RPE 9.5–10.0)")

for bar, row in zip(bars, rep_df.itertuples()):
    height = bar.get_height()
    loss_str = "Baseline" if row.rep == 1 else f"-{row.mean_loss_pct:.1f}%"
    ax3.annotate(f"{height:.1f}%\n({loss_str})",
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 5), textcoords="offset points",
                 ha='center', va='bottom', fontsize=9.5, fontweight='bold')

ax3.set_ylabel("Mean Concentric Velocity\n(% of Rep 1)", fontsize=10)
ax3.set_title("Discrete Intra-Set Velocity Loss (RPE & Proximity to Failure)", fontsize=11, fontweight="bold", loc="left")
ax3.set_ylim(0, 125)
ax3.grid(axis='y', linestyle=":", alpha=0.6)
ax3.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0, framealpha=0.9, fontsize=9)

# Optional auto-save high-res figure without legend clipping
output_plot_path = csv_path.replace(".csv", "_ANALYSIS.png")
plt.savefig(output_plot_path, dpi=300, bbox_inches="tight")
print(f"Analysis plot saved to {output_plot_path}")

plt.show()