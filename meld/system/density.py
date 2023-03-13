from meld.system import scalers
from openmm import unit as u  # type: ignore
import numpy as np  # type: ignore
import copy 
import scipy.ndimage # type: ignore

class DensityManager:
    def __init__(self):
        self.densities = []


    def add_density(self, filename, blur_scaler: scalers.BlurScaler, threshold=None, scale_factor=None):
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
        return density 


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
        density_data = self.map_potential(density_data,threshold,scale_factor) 
        self.blur_scaler = blur_scaler
        if blur_scaler._scaler_key_ == "constant_blur":
            tmp_pot = scipy.ndimage.gaussian_filter(density_data_cp,blur_scaler.blur)
            tmp_pot = np.matrix.flatten(self.map_potential(tmp_pot,threshold,scale_factor))
            self.density_data = np.array([tmp_pot.tolist()]*blur_scaler._num_replicas).astype(np.float64)
        elif blur_scaler._scaler_key_ == "linear_blur":
            density_data = np.matrix.flatten(density_data)
            for i in np.linspace(blur_scaler._min_blur,blur_scaler._max_blur,blur_scaler._num_replicas):
                tmp_pot = scipy.ndimage.gaussian_filter(density_data_cp,i)
                tmp_pot = np.matrix.flatten(self.map_potential(tmp_pot,threshold,scale_factor))
                density_data = np.vstack((density_data,tmp_pot)).astype(np.float64)
            self.density_data = density_data

        self.origin = np.array(origin.value_in_unit(u.nanometer))
        self.voxel_size = np.array(voxel_size.value_in_unit(u.nanometer))
    
    def map_potential(self, map, threshold, scale_factor):
        map_cp = copy.deepcopy(map)
        map = scale_factor * ((map - threshold) / (map.max() - threshold))
        map_where = np.where(map <= 0)
        map_cp = scale_factor * (1 - (map_cp - threshold) / (map_cp.max() - threshold))
        map_cp[map_where[0], map_where[1], map_where[2]] = scale_factor
        return map_cp

