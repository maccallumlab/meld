from meld.system import scalers
from simtk import unit as u  # type: ignore
import numpy as np  # type: ignore


class DensityManager:
    def __init__(self):
        self.densities = []

    def add_density(self, filename, blur_scaler: scalers.BlurScaler):
        try:
            import mrcfile  # type: ignore
        except ImportError:
            print("***")
            print("The mrcfile package must be installed to use density maps.")
            print("***")
            raise

        mrc = mrcfile.open(filename)
        density_data = mrc.data
        origin = mrc.header["origin"].item() * u.angstrom
        voxel_size = mrc.voxel_size.item() * u.angstrom

        density = DensityMap(density_data, origin, voxel_size, blur_scaler)
        self.densities.append(density)
        return len(self.densities) - 1

    def is_valid_id(self, map_id):
        n = len(self.densities)
        return 0 <= map_id < n


class DensityMap:
    def __init__(self, density_data, origin, voxel_size, blur_scaler):
        self.density_data = density_data
        self.nx = density_data.shape[0]
        self.ny = density_data.shape[1]
        self.nz = density_data.shape[2]

        self.origin = np.array(origin.value_in_unit(u.nanometer))
        self.voxel_size = np.array(voxel_size.value_in_unit(u.nanometer))
        self.blur_scaler = blur_scaler
