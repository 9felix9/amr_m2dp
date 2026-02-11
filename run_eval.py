# run_eval.py
import os
import csv
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import precision_recall_curve, average_precision_score


############################

# Running this python script to run the matching algorithm (same as in the matchingAlgo.py) with different test configs. 
# After running the algorithm different statistics are calculated and plots are made for visualization

############################


# ============================================================
# CONFIG (edit me)
# ============================================================

# Sessions you want to evaluate
SESSIONS = [
    {
        "name": "2012-01-08",
        "descriptors_npz": "/Users/felix/Projects/amr_hw3/velodyne_data/2012-01-08/m2dp_descriptors_2012-01-08.npz",
        "groundtruth_csv": "/Users/felix/Projects/amr_hw3/ground_truth/groundtruth_2012-01-08.csv",
    },
    {
        "name": "2012-01-15",
        "descriptors_npz": "/Users/felix/Projects/amr_hw3/velodyne_data/2012-01-15/m2dp_descriptors_2012-01-15.npz",
        "groundtruth_csv": "/Users/felix/Projects/amr_hw3/ground_truth/groundtruth_2012-01-15.csv",
    },
    {
        "name": "2012-01-22",
        "descriptors_npz": "/Users/felix/Projects/amr_hw3/velodyne_data/2012-01-22/m2dp_descriptors_2012-01-22.npz",
        "groundtruth_csv": "/Users/felix/Projects/amr_hw3/ground_truth/groundtruth_2012-01-22.csv",
    },
]

# Evaluation sweeps ("different conditions")
EXCLUSION_FRAMES_LIST = [50]         # e.g. [10, 25, 50, 100]
GT_LOOP_RADIUS_M_LIST = [10.0]       # e.g. [5.0, 10.0, 20.0]
K_NEIGHBORS_LIST = [1_000]           # e.g. [100, 500, 1000]

# Auto scaling
# - set GT_SCALE = 1.0 if x,y,z already in meters
# - set GT_SCALE = 1000.0 if x,y,z in kilometers
GT_SCALE = None  # None = auto-detect (same heuristic as your code)

# Output directories
PLOTS_BASE_DIR = "plots"
RESULTS_DIR = "results"
SAVE_RESULTS_CSV = True

# PR curve plot
SAVE_PR_CURVE = True
PR_DPI = 200

# Trajectory prediction plot (your second plot)
SAVE_PREDICTION_PLOT = True
PRED_DPI = 200

# Threshold for prediction plot
# Uses the SAME concept as your current code:
#   thr = np.percentile(nn_desc_dist, 10)
PRED_THRESHOLD_MODE = "fixed"   # "percentile" | "fixed" | "f1_optimal"
PRED_THRESHOLD_PERCENTILE = 10
PRED_THRESHOLD_FIXED_VALUE = 0.3

# Plot style
PRED_FIGSIZE = (10, 8)
PRED_SCATTER_SIZE = 8
PRED_SHOW_MATCHES = True  # show matched reference points as "x"


# ============================================================
# Your code (logic preserved) – only wrapped for systematic runs
# ============================================================

def load_descriptors_npz(path: str):
    data = np.load(path)
    utimes = data["utimes"].astype(np.int64)
    descriptors = data["descriptors"].astype(np.float32)
    return utimes, descriptors


def load_groundtruth_csv(path: str):
    """
    CSV rows: utime, x, y, z, roll, pitch, yaw
    Some rows may contain NaNs (e.g., first line).
    """
    gt = np.genfromtxt(path, delimiter=",", dtype=np.float64)
    if gt.ndim == 1:
        gt = gt[None, :]

    gt_utime = gt[:, 0].astype(np.int64)
    gt_xyz = gt[:, 1:4].astype(np.float64)

    # Filter rows where x/y/z are finite
    good = np.isfinite(gt_xyz).all(axis=1) & np.isfinite(gt_utime)
    gt_utime = gt_utime[good]
    gt_xyz = gt_xyz[good]

    # Sort by time
    order = np.argsort(gt_utime)
    return gt_utime[order], gt_xyz[order]


def auto_scale_xyz_to_meters(gt_xyz: np.ndarray, gt_scale: float | None):
    """
    Heuristic: if coords are 'small' (campus should be hundreds/thousands in meters),
    then they are likely in kilometers -> multiply by 1000.
    """
    if gt_scale is not None:
        return gt_xyz * float(gt_scale), float(gt_scale)

    # Heuristic: if overall span is < ~20 (units), likely km-scale for NCLT
    span = np.nanmax(gt_xyz, axis=0) - np.nanmin(gt_xyz, axis=0)
    span_norm = np.linalg.norm(span)

    # If span_norm is small, assume kilometers
    if span_norm < 20.0:
        return gt_xyz * 1000.0, 1000.0
    return gt_xyz * 1.0, 1.0


def align_pose_to_scans(scan_utimes: np.ndarray, gt_utimes: np.ndarray, gt_xyz_m: np.ndarray):
    idx = np.searchsorted(gt_utimes, scan_utimes, side="left")

    idx0 = np.clip(idx - 1, 0, len(gt_utimes) - 1)
    idx1 = np.clip(idx,     0, len(gt_utimes) - 1)

    t0 = gt_utimes[idx0]
    t1 = gt_utimes[idx1]

    pick1 = np.abs(scan_utimes - t1) < np.abs(scan_utimes - t0)
    chosen = np.where(pick1, idx1, idx0)
    return gt_xyz_m[chosen]


def nearest_neighbor_excluding_window(descriptors: np.ndarray, exclusion_frames: int, k_neighbors: int):
    N = descriptors.shape[0]
    k = min(k_neighbors, N)

    nn = NearestNeighbors(n_neighbors=k, metric="euclidean", algorithm="auto")
    nn.fit(descriptors)

    dists, inds = nn.kneighbors(descriptors, return_distance=True)

    nn_idx = np.full(N, -1, dtype=np.int64)
    nn_dist = np.full(N, np.inf, dtype=np.float64)

    for i in range(N):
        for d, j in zip(dists[i], inds[i]):
            if abs(int(j) - i) > exclusion_frames:
                nn_idx[i] = int(j)
                nn_dist[i] = float(d)
                break

    missing = np.where(nn_idx < 0)[0]
    if len(missing) > 0:
        raise RuntimeError(
            f"Could not find valid NN outside exclusion window for {len(missing)} frames. "
            f"Increase K_NEIGHBORS (currently {k_neighbors}) or reduce EXCLUSION_FRAMES."
        )

    return nn_idx, nn_dist


def compute_gt_labels(scan_xyz_m: np.ndarray, nn_idx: np.ndarray, gt_radius_m: float):
    diffs = scan_xyz_m - scan_xyz_m[nn_idx]
    spatial_dist_m = np.linalg.norm(diffs, axis=1)
    y_true = (spatial_dist_m < gt_radius_m).astype(np.int32)
    return y_true, spatial_dist_m


def compute_pr_from_nn_dist(nn_desc_dist: np.ndarray, y_true: np.ndarray):
    scores = -nn_desc_dist.astype(np.float64)
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    ap = average_precision_score(y_true, scores)
    return precision, recall, thresholds, ap


# ============================================================
# Plot saving utilities (no algorithm change)
# ============================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_pr_curve_plot(session_dir: str, session_name: str, exclusion_frames: int, gt_radius_m: float, k_neighbors: int,
                       recall: np.ndarray, precision: np.ndarray, ap: float):
    if not SAVE_PR_CURVE:
        return

    ensure_dir(session_dir)

    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(
        f"PR Curve | {session_name} | excl=±{exclusion_frames} | GT<{gt_radius_m}m | k={k_neighbors} | AP={ap:.3f}"
    )
    plt.grid(True)

    fname = f"pr_curve_excl{exclusion_frames}_gt{gt_radius_m:g}_k{k_neighbors}.png"
    outpath = os.path.join(session_dir, fname)
    plt.savefig(outpath, dpi=PR_DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved PR curve: {outpath}")


def determine_prediction_threshold(nn_desc_dist: np.ndarray, best_f1_thr: float):
    if PRED_THRESHOLD_MODE == "percentile":
        return float(np.percentile(nn_desc_dist, PRED_THRESHOLD_PERCENTILE))
    if PRED_THRESHOLD_MODE == "fixed":
        return float(PRED_THRESHOLD_FIXED_VALUE)
    if PRED_THRESHOLD_MODE == "f1_optimal":
        return float(best_f1_thr)
    raise ValueError(f"Unknown PRED_THRESHOLD_MODE: {PRED_THRESHOLD_MODE}")


def save_prediction_plot(session_dir: str, session_name: str, exclusion_frames: int, gt_radius_m: float, k_neighbors: int,
                         scan_xyz_m: np.ndarray, nn_idx: np.ndarray, nn_desc_dist: np.ndarray, threshold: float):
    if not SAVE_PREDICTION_PLOT:
        return

    ensure_dir(session_dir)

    pred = nn_desc_dist < threshold
    xy = scan_xyz_m[:, :2]
    x, y = xy[:, 1], xy[:, 0]

    plt.figure(figsize=PRED_FIGSIZE)
    plt.plot(x, y, linewidth=1)
    plt.scatter(x[pred], y[pred], s=PRED_SCATTER_SIZE, alpha=0.7, label="Query (predicted LC)")

    if PRED_SHOW_MATCHES:
        matched = nn_idx[pred]
        plt.scatter(x[matched], y[matched], s=PRED_SCATTER_SIZE, alpha=0.7, marker="x", label="Matched reference")

    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title(f"Predicted loop-closure points | {session_name} | thr={threshold:.4f}")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()

    fname = (
        f"predicted_points_excl{exclusion_frames}_gt{gt_radius_m:g}_k{k_neighbors}_"
        f"thr_{PRED_THRESHOLD_MODE}_{threshold:.4f}.png"
    )
    outpath = os.path.join(session_dir, fname)
    plt.savefig(outpath, dpi=PRED_DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved prediction plot: {outpath}")


# ============================================================
# Single run (session + condition)
# ============================================================

def run_one(session_name: str, descriptors_npz: str, groundtruth_csv: str,
            exclusion_frames: int, gt_loop_radius_m: float, k_neighbors: int):

    # Load
    scan_utimes, descriptors = load_descriptors_npz(descriptors_npz)
    gt_utimes, gt_xyz = load_groundtruth_csv(groundtruth_csv)
    gt_xyz_m, used_scale = auto_scale_xyz_to_meters(gt_xyz, GT_SCALE)

    scan_xyz_m = align_pose_to_scans(scan_utimes, gt_utimes, gt_xyz_m)

    print("\n============================================================")
    print(f"SESSION: {session_name} | excl=±{exclusion_frames} | GT<{gt_loop_radius_m}m | k={k_neighbors}")
    print("Descriptors:", descriptors.shape)
    print("Scan utimes:", scan_utimes.shape)
    print("GT poses:", gt_xyz.shape, f"(scaled by {used_scale} -> meters)")
    print("Aligned scan poses:", scan_xyz_m.shape)

    # 1-NN matching with exclusion window
    nn_idx, nn_desc_dist = nearest_neighbor_excluding_window(
        descriptors=descriptors,
        exclusion_frames=exclusion_frames,
        k_neighbors=k_neighbors,
    )

    # GT labels for those matches (loop closure if spatial < gt_loop_radius_m)
    y_true, spatial_dist_m = compute_gt_labels(
        scan_xyz_m=scan_xyz_m,
        nn_idx=nn_idx,
        gt_radius_m=gt_loop_radius_m,
    )

    precision, recall, thresholds, ap = compute_pr_from_nn_dist(nn_desc_dist, y_true)

    # F1-score computation (same as your code)
    dist_thresholds = -thresholds
    f1_scores = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)

    best_idx = int(np.argmax(f1_scores))
    best_f1 = float(f1_scores[best_idx])
    best_thr = float(dist_thresholds[best_idx])
    best_prec = float(precision[best_idx])
    best_rec = float(recall[best_idx])

    print(f"Best F1-score: {best_f1:.4f}")
    print(f"  at descriptor-distance threshold: {best_thr:.4f}")
    print(f"  Precision: {best_prec:.4f}, Recall: {best_rec:.4f}")
    print(f"GT positives (<{gt_loop_radius_m}m): {int(y_true.sum())} / {len(y_true)}")
    print(f"Average Precision (AP): {float(ap):.4f}")

    # Save plots
    session_dir = os.path.join(PLOTS_BASE_DIR, session_name)
    save_pr_curve_plot(
        session_dir=session_dir,
        session_name=session_name,
        exclusion_frames=exclusion_frames,
        gt_radius_m=gt_loop_radius_m,
        k_neighbors=k_neighbors,
        recall=recall,
        precision=precision,
        ap=float(ap),
    )

    pred_thr = determine_prediction_threshold(nn_desc_dist, best_f1_thr=best_thr)
    save_prediction_plot(
        session_dir=session_dir,
        session_name=session_name,
        exclusion_frames=exclusion_frames,
        gt_radius_m=gt_loop_radius_m,
        k_neighbors=k_neighbors,
        scan_xyz_m=scan_xyz_m,
        nn_idx=nn_idx,
        nn_desc_dist=nn_desc_dist,
        threshold=pred_thr,
    )

    # Return for CSV logging
    return {
        "session": session_name,
        "descriptors_npz": descriptors_npz,
        "groundtruth_csv": groundtruth_csv,
        "exclusion_frames": exclusion_frames,
        "gt_loop_radius_m": gt_loop_radius_m,
        "k_neighbors": k_neighbors,
        "gt_scale_used": used_scale,
        "N": int(descriptors.shape[0]),
        "gt_positives": int(y_true.sum()),
        "ap": float(ap),
        "best_f1": best_f1,
        "best_f1_threshold": best_thr,
        "best_precision": best_prec,
        "best_recall": best_rec,
        "prediction_plot_threshold_mode": PRED_THRESHOLD_MODE,
        "prediction_plot_threshold_value": float(pred_thr),
    }


# ============================================================
# CSV logging
# ============================================================

def write_results_csv(rows: list[dict], out_csv_path: str):
    ensure_dir(os.path.dirname(out_csv_path))

    # fixed column order for clean tables
    fieldnames = [
        "session",
        "exclusion_frames",
        "gt_loop_radius_m",
        "k_neighbors",
        "gt_scale_used",
        "N",
        "gt_positives",
        "ap",
        "best_f1",
        "best_f1_threshold",
        "best_precision",
        "best_recall",
        "prediction_plot_threshold_mode",
        "prediction_plot_threshold_value",
        "descriptors_npz",
        "groundtruth_csv",
    ]

    with open(out_csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"\nSaved results CSV: {out_csv_path}")


# ============================================================
# Main
# ============================================================

def main():
    ensure_dir(PLOTS_BASE_DIR)
    ensure_dir(RESULTS_DIR)

    all_rows = []

    for sess in SESSIONS:
        for excl in EXCLUSION_FRAMES_LIST:
            for gt_r in GT_LOOP_RADIUS_M_LIST:
                for k in K_NEIGHBORS_LIST:
                    row = run_one(
                        session_name=sess["name"],
                        descriptors_npz=sess["descriptors_npz"],
                        groundtruth_csv=sess["groundtruth_csv"],
                        exclusion_frames=excl,
                        gt_loop_radius_m=gt_r,
                        k_neighbors=k,
                    )
                    all_rows.append(row)

    if SAVE_RESULTS_CSV:

        # --- Build readable filename from config ---
        session_names = [s["name"] for s in SESSIONS]

        if len(session_names) == 1:
            session_part = f"sessions-{session_names[0]}"
        else:
            session_part = f"sessions-{len(session_names)}"

        excl_part = f"excl{'-'.join(map(str, EXCLUSION_FRAMES_LIST))}"
        gt_part = f"gt{'-'.join(map(lambda x: str(int(x)) if float(x).is_integer() else str(x), GT_LOOP_RADIUS_M_LIST))}"
        k_part = f"k{'-'.join(map(str, K_NEIGHBORS_LIST))}"

        filename = f"results_{session_part}_{excl_part}_{gt_part}_{k_part}.csv"

        out_csv = os.path.join(RESULTS_DIR, filename)

        write_results_csv(all_rows, out_csv)


if __name__ == "__main__":
    main()
