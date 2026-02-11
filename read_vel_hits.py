import struct
import numpy as np

# this reader unpacks the coordinates from each velodyne measurement - based on the provided python code by the nclt dataset website under :
# https://robots.engin.umich.edu/nclt/index.html


class VelodyneSyncReader:
    def read(self, bin_path):
        points = []

        with open(bin_path, "rb") as f:
            while True:
                hit = f.read(8)
                if hit == b"":
                    break

                x_s, y_s, z_s, i, l = struct.unpack("<HHHBB", hit)

                # exakt dieselbe Konvertierung wie vorher
                x = x_s * 0.005 - 100.0
                y = y_s * 0.005 - 100.0
                z = z_s * 0.005 - 100.0

                points.append([x, y, z])

        return np.array(points)

 
if __name__ == "__main__":
    VelodyneSyncReader().read("/Users/felix/Projects/amr_hw3/velodyne_data/2012-01-22/velodyne_sync")