import struct
import numpy as np


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
