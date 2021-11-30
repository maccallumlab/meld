from meld.system import scalers
from simtk import unit as u  # type: ignore
import numpy as np  # type: ignore
import copy

class DensityManager:
    def __init__(self):
        self.densities = []


    def add_density(self, filename, blur_scaler: scalers.BlurScaler, scale_factor=None, threshold=None):
        try:
            import mrcfile  # type: ignore
        except ImportError:
            print("***")
            print("The mrcfile package must be installed to use density maps.")
            print("***")
            raise
        scale_factor = 0.3 if scale_factor is None else scale_factor
        threshold = 0 if threshold is None else threshold
        mrc = mrcfile.open(filename)
        density_data = mrc.data
        origin = mrc.header["origin"].item() * u.angstrom
        voxel_size = mrc.voxel_size.item() * u.angstrom

        density = DensityMap(density_data, origin, voxel_size, blur_scaler,scale_factor,threshold)
        self.densities.append(density)
        return len(self.densities) - 1

    def is_valid_id(self, map_id):
        n = len(self.densities)
        return 0 <= map_id < n


class DensityMap:
    def __init__(
        self, 
        density_data, 
        origin, 
        voxel_size, 
        blur_scaler,
        scale_factor,
        threshold
    ):  
        self.scale_factor = scale_factor
        self.threshold = threshold
        self.nx = density_data.shape[2]
        self.ny = density_data.shape[1]
        self.nz = density_data.shape[0]
        density_data_cp = copy.deepcopy(density_data)
        density_data = self.scale_factor * ((density_data - self.threshold) / (density_data.max() - self.threshold))
        tmp_index = np.where(density_data <= 0)
        density_data_cp = self.scale_factor * (1 - (density_data_cp - self.threshold) / (density_data_cp.max() - self.threshold))
        density_data_cp[tmp_index[0], tmp_index[1], tmp_index[2]] = self.scale_factor

        self.density_data = density_data_cp

        self.origin = np.array(origin.value_in_unit(u.nanometer))
        self.voxel_size = np.array(voxel_size.value_in_unit(u.nanometer))
        self.blur_scaler = blur_scaler
