from scipy import ndimage
import numpy as np
import string
import pandas as pd
import warnings
class NormaliseHeight:
    def __init__(self, pc):
        self.pc = pc
        warnings.filterwarnings('ignore', 'All-NaN slice encountered', RuntimeWarning, 'numpy.lib.nanfunctions')
    def voxelise(self):
        self.pc.loc[:, 'xx'] = self.pc.x // 0.5 * 0.5
        self.pc.loc[:, 'yy'] = self.pc.y // 0.5 * 0.5
        code = lambda: ''.join(np.random.choice([x for x in string.ascii_letters], size=8))
        xD = {x: code() for x in self.pc.xx.unique()}
        yD = {y: code() for y in self.pc.yy.unique()}
        self.pc.loc[:, 'VX'] = self.pc.xx.map(xD) + self.pc.yy.map(yD)
        return self.pc
    def height_normalise(self):
        """ 
        This function will generate a Digital Terrain Model (dtm) based on the terrain labelled points and height normalise.
        """
        self.voxelise()
        VX_map = self.pc.loc[~self.pc.VX.duplicated()][['xx', 'yy', 'VX']]
        ground = self.pc.loc[self.pc.label == 2].copy()
        ground.loc[:, 'zmin'] = ground.groupby('VX').z.transform(np.nanmedian).copy() 
        ground = ground.loc[ground.z == ground.zmin]
        ground = ground.loc[~ground.VX.duplicated()]
        X, Y = np.meshgrid(np.arange(self.pc.xx.min(), self.pc.xx.max() + 0.5, 0.5),
                            np.arange(self.pc.yy.min(), self.pc.yy.max() + 0.5, 0.5))
        ground_arr = pd.DataFrame(data=np.vstack([X.flatten(), Y.flatten()]).T, columns=['xx', 'yy']) 
        ground_arr = pd.merge(ground_arr, VX_map, on=['xx', 'yy'], how='outer') # map VX to ground_arr
        ground_arr = pd.merge(ground[['z', 'VX']], ground_arr, how='right', on=['VX']) # map z to ground_arr
        ground_arr.sort_values(['xx', 'yy'], inplace=True)
        # loop over incresing size of window until no cell are nan
        ground_arr.loc[:, 'ZZ'] = np.nan
        size = 3 
        while np.any(np.isnan(ground_arr.ZZ)):
            ground_arr['ZZ'] = ndimage.generic_filter(ground_arr.z.values.reshape(*X.shape), lambda z: np.nanmedian(z), size=size).flatten()
            size += 2
        # apply to all points   
        MAP = ground_arr.set_index('VX').ZZ.to_dict()
        self.pc.loc[:, 'n_z'] = self.pc.z - self.pc.VX.map(MAP)  
        return self.pc
