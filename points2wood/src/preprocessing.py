import torch
from torch_geometric.nn import voxel_grid
from torch_geometric.nn.pool.consecutive import consecutive_cluster
import glob
from src.io import load_file, save_file
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

class Voxelise:
    def __init__(self, pos, vxpath, minpoints=512, maxpoints=8192, gridsize=[1.0,2.0], pointspacing=0.01):
        self.pos = pos
        self.vxpath = vxpath
        self.minpoints = minpoints
        self.maxpoints = maxpoints
        self.gridsize = gridsize
        self.pointspacing = min(self.gridsize) / 100.0#pointspacing

    def quantile_normalize_reflectance(self):
        reflectance_tensor = self.pos[:, 3].view(-1)
        _, indices = torch.sort(reflectance_tensor)
        ranks = torch.argsort(indices)
        empirical_quantiles = (ranks.float() + 1) / (len(ranks) + 1)
        normalized_reflectance = torch.erfinv(2 * empirical_quantiles - 1) * torch.sqrt(torch.tensor(2.0)).to(reflectance_tensor.device)
        min_val = normalized_reflectance.min()
        max_val = normalized_reflectance.max()
        scaled_reflectance = 2 * (normalized_reflectance - min_val) / (max_val - min_val) - 1
        #scaled_reflectance = (normalized_reflectance - min_val) / (max_val - min_val)#0-1 scaling
        return scaled_reflectance
    
    def downsample(self):
        voxelised = voxel_grid(self.pos, self.pointspacing)
        _, idx = consecutive_cluster(voxelised)
        return self.pos[idx]
        
    def grid(self):
        indices_list = []
        for size in self.gridsize:
            # voxelised = voxel_grid(self.pos, size / 100.0)
            # _, downsampled = consecutive_cluster(voxelised)
            # downsampled = self.pos[downsampled]

            voxelised = voxel_grid(self.pos, size).to('cpu')
            for vx in torch.unique(voxelised):
                voxel = (voxelised == vx).nonzero(as_tuple=True)[0]
                if voxel.size(0) < self.minpoints:
                    continue
                indices_list.append(voxel)
        return indices_list
    
    def write_voxels(self):

        if not torch.all(self.pos[:, 3] == 0):
            self.pos[:, 3] = self.quantile_normalize_reflectance()

        #self.pos = self.downsample()
        voxels = self.grid()
        
        file_counter = len(glob.glob(os.path.join(self.vxpath, 'voxel_*.pt')))
        self.pos = self.pos.detach().clone().to('cpu')
        for _, voxel_indices in enumerate(tqdm(voxels, desc='Writing voxels')):
            
            if not torch.all(self.pos[:, 3] == 0):
                weight = self.pos[voxel_indices, 3] - min(self.pos[voxel_indices, 3])
                mask = ~(torch.isnan(weight) | torch.isinf(weight) | (weight < 0))
                voxel_indices, weight = voxel_indices[mask], weight[mask]

            if voxel_indices.size(0) == 0: continue  # Skip if no valid indices

            if voxel_indices.size(0) > self.maxpoints:
                if not torch.all(self.pos[:, 3] == 0):
                    voxel_indices = voxel_indices[torch.multinomial(weight, self.maxpoints)]
                else: 
                    voxel_indices = voxel_indices[torch.randint(0, voxel_indices.size(0), (self.maxpoints,))]

            voxel = self.pos[voxel_indices]
            voxel = voxel[~torch.isnan(voxel).any(dim=1)]
            torch.save(voxel, os.path.join(self.vxpath, f'voxel_{file_counter}.pt'))
            #np.savetxt(os.path.join(self.vxpath, f'voxel_{file_counter}.txt'), voxel.cpu().numpy(), fmt='%.6f')
            file_counter += 1

def preprocess(args):
    Voxelise(torch.tensor(args.pc.values, dtype=torch.float32).to('cuda'),
             vxpath=args.vxfile, minpoints=args.min_pts, maxpoints=args.max_pts,
             pointspacing=args.resolution, gridsize = args.grid_size).write_voxels()







# '''LOAD POINT CLOUD AND CONVERT TO TENSOR'''
# cloud, _ = load_file('/home/harryowen/Desktop/test.ply', additional_headers=True)
# pos = torch.tensor(cloud.values[:,:4], dtype=torch.float32).to('cuda')  # Specify dtype as needed
# #index_tensor = torch.arange(pos.size(0), dtype=torch.int64).to('cuda')
# voxeliser = Voxelise(pos, '/home/harryowen/Desktop/tmp/', gridsize=[1.0,4.0])
# voxeliser.write_voxels()


    # def grid(self):
    #     indices_list = []
    #     for size in self.gridsize:
    #         voxelised = voxel_grid(self.pos, size).to('cpu')
    #         voxels = torch.unique(voxelised)

    #         while True:
    #             # List to hold indices of voxels that need to be merged
    #             to_merge = []

    #             for vx in voxels:
    #                 voxel_indices = (voxelised == vx).nonzero(as_tuple=True)[0]
    #                 if voxel_indices.size(0) < self.minpoints:
    #                     to_merge.append(voxel_indices)
    #                 else:
    #                     indices_list.append(voxel_indices)
                
    #             if not to_merge:
    #                 break  # Exit the loop if no voxels need merging

    #             # Merge small voxels with their nearest neighbor
    #             vx_pos = torch.stack([self.pos[indices].mean(dim=0) for indices in to_merge])
    #             kdtree = cKDTree(vx_pos.cpu())
    #             _, indices = kdtree.query(vx_pos.cpu(), k=2)

    #             for i, voxels in enumerate(to_merge):

    #                 nearest_neighbor = to_merge[indices[i,1]]

    #                 # Combine the indices
    #                 combined_indices = torch.cat((voxels, nearest_neighbor))

    #                 # Update voxelised with new indices
    #                 voxelised[combined_indices] = nearest_neighbor.item()  # Update to the nearest neighbor's voxel value

    #             voxels = torch.unique(voxelised)

    #     return indices_list