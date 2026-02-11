import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import precision_recall_curve, average_precision_score


############################

# Running this python script compares the descriptor vectors to find potential loop closure locations. 
# The vectors are compared by the L2 norm (euclidean distance). 

############################

# --------------------------
# CONFIG
# --------------------------
DESCRIPTORS_NPZ = "/Users/felix/Projects/amr_hw3/velodyne_data/2012-01-08/m2dp_descriptors_2012-01-08.npz"
GROUNDTRUTH_CSV = "/Users/felix/Projects/amr_hw3/ground_truth/groundtruth_2012-01-08.csv"

EXCLUSION_FRAMES = 50        # +/- 50 frames
GT_LOOP_RADIUS_M = 10.0      # GT loop closure if < 10ms
K_NEIGHBORS = 1_000          # retrieve top-k neighbors, then pick first outside exclusion window

# If you want to force unit scaling:
#   - set GT_SCALE = 1.0 if x,y,z already in meters
#   - set GT_SCALE = 1000.0 if x,y,z in kilometers
GT_SCALE = None  # None = auto-detect


# --------------------------
# Load descriptors
# --------------------------
def load_descriptors_npz(path: str):
    data = np.load(path)
    utimes = data["utimes"].astype(np.int64)
    descriptors = data["descriptors"].astype(np.float32)
    return utimes, descriptors


# --------------------------
# Load groundtruth CSV robustly
# --------------------------
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


# --------------------------
# Align GT poses to scan times (nearest timestamp)
# --------------------------
def align_pose_to_scans(scan_utimes: np.ndarray, gt_utimes: np.ndarray, gt_xyz_m: np.ndarray):
    idx = np.searchsorted(gt_utimes, scan_utimes, side="left")

    idx0 = np.clip(idx - 1, 0, len(gt_utimes) - 1)
    idx1 = np.clip(idx,     0, len(gt_utimes) - 1)

    t0 = gt_utimes[idx0]
    t1 = gt_utimes[idx1]

    pick1 = np.abs(scan_utimes - t1) < np.abs(scan_utimes - t0)
    chosen = np.where(pick1, idx1, idx0)
    return gt_xyz_m[chosen]


# --------------------------
# Matching: NN in descriptor space excluding +/- EXCLUSION_FRAMES
# --------------------------
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


# --------------------------
# GT label for each (i -> nn_idx[i]) match
# --------------------------
def compute_gt_labels(scan_xyz_m: np.ndarray, nn_idx: np.ndarray, gt_radius_m: float):
    diffs = scan_xyz_m - scan_xyz_m[nn_idx]
    spatial_dist_m = np.linalg.norm(diffs, axis=1)
    y_true = (spatial_dist_m < gt_radius_m).astype(np.int32)
    return y_true, spatial_dist_m


# --------------------------
# PR curve by varying descriptor-distance threshold
# "predict positive if nn_desc_dist < threshold"
# sklearn expects higher score => more positive => use score = -dist
# --------------------------
def compute_pr_from_nn_dist(nn_desc_dist: np.ndarray, y_true: np.ndarray):
    scores = -nn_desc_dist.astype(np.float64)
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    ap = average_precision_score(y_true, scores)
    return precision, recall, thresholds, ap


# --------------------------
# RUN
# --------------------------
scan_utimes, descriptors = load_descriptors_npz(DESCRIPTORS_NPZ)
gt_utimes, gt_xyz = load_groundtruth_csv(GROUNDTRUTH_CSV)
gt_xyz_m, used_scale = auto_scale_xyz_to_meters(gt_xyz, GT_SCALE)

scan_xyz_m = align_pose_to_scans(scan_utimes, gt_utimes, gt_xyz_m)

print("Descriptors:", descriptors.shape)
print("Scan utimes:", scan_utimes.shape)
print("GT poses:", gt_xyz.shape, f"(scaled by {used_scale} -> meters)")
print("Aligned scan poses:", scan_xyz_m.shape)

# 1-NN matching with exclusion window
nn_idx, nn_desc_dist = nearest_neighbor_excluding_window(
    descriptors=descriptors,
    exclusion_frames=EXCLUSION_FRAMES,
    k_neighbors=K_NEIGHBORS,
)

# GT labels for those matches (loop closure if spatial < 10m)
y_true, spatial_dist_m = compute_gt_labels(
    scan_xyz_m=scan_xyz_m,
    nn_idx=nn_idx,
    gt_radius_m=GT_LOOP_RADIUS_M,
)

precision, recall, thresholds, ap = compute_pr_from_nn_dist(nn_desc_dist, y_true)

# --------------------------
# F1-score computation
# --------------------------
# precision_recall_curve returns:
#   precision: length = len(thresholds) + 1
#   recall:    length = len(thresholds) + 1
#   thresholds are in score-space (score = -distance)

# Convert score-thresholds to distance-thresholds
dist_thresholds = -thresholds

# Compute F1 for all valid points (exclude last PR point without threshold)
f1_scores = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)

best_idx = np.argmax(f1_scores)
best_f1 = f1_scores[best_idx]
best_thr = dist_thresholds[best_idx]
best_prec = precision[best_idx]
best_rec = recall[best_idx]

print(f"Best F1-score: {best_f1:.4f}")
print(f"  at descriptor-distance threshold: {best_thr:.4f}")
print(f"  Precision: {best_prec:.4f}, Recall: {best_rec:.4f}")


print(f"GT positives (<{GT_LOOP_RADIUS_M}m): {int(y_true.sum())} / {len(y_true)}")
print(f"Average Precision (AP): {ap:.4f}")

# Plot PR curve
plt.figure()
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title(
    f"PR Curve (intra-session) | excl=±{EXCLUSION_FRAMES} | GT<{GT_LOOP_RADIUS_M}m | AP={ap:.3f}"
)
plt.grid(True)
plt.show()

# Optional: sanity print a few samples
rng = np.random.default_rng(0)
sample = rng.choice(len(scan_utimes), size=min(10, len(scan_utimes)), replace=False)
print("\nSample matches (i -> j):")
for i in sample:
    j = nn_idx[i]
    print(
        f"i={i:6d} -> j={j:6d}  desc_dist={nn_desc_dist[i]:.4f}  spatial_dist={spatial_dist_m[i]:.2f}m  GT={y_true[i]}"
    )


import matplotlib.pyplot as plt
import numpy as np

def plot_predicted_points(scan_xyz_m, nn_idx, nn_desc_dist, threshold, show_matches=True):
    pred = nn_desc_dist < threshold
    xy = scan_xyz_m[:, :2]
    x, y = xy[:, 1], xy[:, 0]

    plt.figure(figsize=(10, 8))
    plt.plot(x, y, linewidth=1)
    plt.scatter(x[pred], y[pred], s=8, alpha=0.7)  # predicted loop-closure query points

    if show_matches:
        matched = nn_idx[pred]
        plt.scatter(x[matched], y[matched], s=8, alpha=0.7, marker="x")

    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title(f"Predicted loop-closure points (thr={threshold:.4f})")
    plt.axis("equal")
    plt.grid(True)
    plt.show()

thr = np.percentile(nn_desc_dist, 10)
plot_predicted_points(scan_xyz_m, nn_idx, nn_desc_dist, thr, show_matches=True)
