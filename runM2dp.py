import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import glob
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

from read_vel_hits import VelodyneSyncReader
from M2DP.m2dp import M2DP

session_velodyne_sync_dir = (
    "/Users/felix/Projects/amr_hw3/velodyne_data/2012-01-08/velodyne_sync"
)
save_npz_path = (
    "/Users/felix/Projects/amr_hw3/velodyne_data/2012-01-08/"
    "m2dp_descriptors_2012-01-08.npz"
)

_reader = None
_m2dp = None

def init_worker():
    global _reader, _m2dp
    _reader = VelodyneSyncReader()
    _m2dp = M2DP()

def process_one(fp: str):
    utime = int(os.path.splitext(os.path.basename(fp))[0])
    points = _reader.read(fp)
    descriptor, _ = _m2dp.runM2DP(points)
    descriptor = np.asarray(descriptor, dtype=np.float32).ravel()
    return utime, descriptor

def main():
    bin_files = sorted(glob.glob(os.path.join(session_velodyne_sync_dir, "*.bin")))
    print(f"Found {len(bin_files)} velodyne scans.")

    max_workers = 4
    chunksize = 50
    ctx = mp.get_context("spawn")

    results = []
    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=ctx,
        initializer=init_worker,
    ) as ex:
        for idx, (utime, desc) in enumerate(ex.map(process_one, bin_files, chunksize=chunksize)):
            results.append((utime, desc))
            if idx % 500 == 0:
                print(f"[{idx}/{len(bin_files)}] utime={utime} desc_dim={desc.shape[0]}")

    results.sort(key=lambda x: x[0])
    utimes = np.array([u for u, _ in results], dtype=np.int64)
    descriptors = np.vstack([d for _, d in results]).astype(np.float32)

    np.savez_compressed(save_npz_path, utimes=utimes, descriptors=descriptors)
    print("Saved:", save_npz_path, utimes.shape, descriptors.shape)

if __name__ == "__main__":
    main()
