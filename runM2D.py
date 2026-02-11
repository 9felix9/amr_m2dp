import os
import time
import glob
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor


############################

# If this script is executed the M2DP algorithm calculated the descriptor vectors of all pointclouds of the given session. 
# As this one is a highly computational task the multiprocessing and pool executer packages from python are used to make the execution run parallel
# results are saved as npz files - efficient numpy files

############################

# ============================================================
# Toggle: Threads begrenzen (FAST) vs nicht begrenzen (SLOW)
# ============================================================
LIMIT_BLAS_THREADS = True

if LIMIT_BLAS_THREADS:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"   
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np

# Optional diagnostics libs
try:
    import psutil
except ImportError:
    psutil = None

try:
    from threadpoolctl import threadpool_info
except ImportError:
    threadpool_info = None

from read_vel_hits import VelodyneSyncReader
from M2DP.m2dp import M2DP

session_velodyne_sync_dir = (
    "/Users/felix/Projects/amr_hw3/velodyne_data/2012-01-22/velodyne_sync"
)

save_npz_path = (
    "/Users/felix/Projects/amr_hw3/velodyne_data/2012-01-22/"
    "m2dp_descriptors_2012-01-22.npz"
)

_reader = None
_m2dp = None

def log_startup_state(prefix: str):
    pid = os.getpid()
    ppid = os.getppid()
    cpu_count = os.cpu_count()

    print(f"{prefix} PID={pid} PPID={ppid} cpu_count={cpu_count}")

    # ENV variables that control threading
    env_keys = [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]
    env_dump = {k: os.environ.get(k) for k in env_keys}
    print(f"{prefix} Thread ENV: {env_dump}")

    # Show numpy build / linked BLAS/LAPACK
    print(f"{prefix} NumPy config (BLAS/LAPACK links) follows:")
    np.show_config()

    # Threadpoolctl (most direct way to see actual threadpool settings)
    if threadpool_info is not None:
        info = threadpool_info()
        print(f"{prefix} threadpoolctl info:")
        for lib in info:
            # Typical keys: internal_api, num_threads, prefix, filepath, version
            print(f"{prefix}  - {lib.get('internal_api')} "
                  f"num_threads={lib.get('num_threads')} "
                  f"prefix={lib.get('prefix')} version={lib.get('version')}")
    else:
        print(f"{prefix} threadpoolctl not installed (pip install threadpoolctl).")

    # Basic system load snapshot
    if psutil is not None:
        proc = psutil.Process(pid)
        print(f"{prefix} RSS memory MB={proc.memory_info().rss / 1e6:.1f}")
    else:
        print(f"{prefix} psutil not installed (pip install psutil).")

def init_worker():
    global _reader, _m2dp
    _reader = VelodyneSyncReader()
    _m2dp = M2DP()
    log_startup_state(prefix="[worker-init]")

def process_one(fp: str):
    """
    Returns timing breakdown to see where time goes.
    """
    utime = int(os.path.splitext(os.path.basename(fp))[0])

    t0 = time.perf_counter()
    points = _reader.read(fp)
    t1 = time.perf_counter()

    descriptor, _ = _m2dp.runM2DP(points)
    t2 = time.perf_counter()

    descriptor = np.asarray(descriptor, dtype=np.float32).ravel()

    io_ms = (t1 - t0) * 1000.0
    m2dp_ms = (t2 - t1) * 1000.0
    total_ms = (t2 - t0) * 1000.0

    return utime, descriptor, points.shape[0], io_ms, m2dp_ms, total_ms

def main():
    log_startup_state(prefix="[main]")

    bin_files = sorted(glob.glob(os.path.join(session_velodyne_sync_dir, "*.bin")))
    print(f"[main] Found {len(bin_files)} velodyne scans.")

    # For quick diagnosis, you can temporarily cap:
    # bin_files = bin_files[:2000]

    max_workers = 10
    chunksize = 50
    ctx = mp.get_context("spawn")

    results = []
    t_start = time.time()

    # rolling stats
    io_times = []
    m2dp_times = []
    total_times = []

    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=ctx,
        initializer=init_worker,
    ) as ex:
        for idx, (utime, desc, npts, io_ms, m2dp_ms, total_ms) in enumerate(
            ex.map(process_one, bin_files, chunksize=chunksize)
        ):
            results.append((utime, desc))
            io_times.append(io_ms)
            m2dp_times.append(m2dp_ms)
            total_times.append(total_ms)

            if idx > 0 and idx % 500 == 0:
                elapsed = time.time() - t_start
                rate = idx / elapsed
                print(
                    f"[main] {idx}/{len(bin_files)} done | "
                    f"rate={rate:.2f} scans/s | "
                    f"median_io={np.median(io_times):.1f}ms | "
                    f"median_m2dp={np.median(m2dp_times):.1f}ms | "
                    f"median_total={np.median(total_times):.1f}ms | " 
                    f"time_elapsed_total={elapsed:.1f}s"
                )

    results.sort(key=lambda x: x[0])
    utimes = np.array([u for u, _ in results], dtype=np.int64)
    descriptors = np.vstack([d for _, d in results]).astype(np.float32)

    np.savez_compressed(save_npz_path, utimes=utimes, descriptors=descriptors)

    print("[main] Saved:")
    print("  file:", save_npz_path)
    print("  utimes:", utimes.shape)
    print("  descriptors:", descriptors.shape)

if __name__ == "__main__":
    main()
