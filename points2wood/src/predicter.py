from src.model import Net

import os
os.environ['OMP_NUM_THREADS'] = str(os.cpu_count()-1)
import sys
import pandas as pd
import numpy as np
from pykdtree.kdtree import KDTree
from tqdm.auto import tqdm
import torch
from abc import ABC
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from src.io import save_file
from utils.height_normalisation import NormaliseHeight
from numba import jit, prange, set_num_threads
import glob

import warnings
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

set_num_threads(30)

sys.setrecursionlimit(10 ** 8) # Can be necessary for dealing with large point clouds.

class TestingDataset(Dataset, ABC):
    def __init__(self, voxels, max_pts, device, in_memory=False):
        if not voxels:
            raise ValueError("The 'voxels' parameter cannot be empty.")
        self.voxels = voxels
        self.keys = sorted(glob.glob(os.path.join(voxels, '*.pt')))
        self.device = device
        self.max_pts = max_pts
        self.reflectance_index = 3
        self.sampleweights_index = 4

    def __len__(self):
        return len(self.keys)  # Use the cached keys

    def __getitem__(self, index):

        point_cloud = torch.load(self.keys[index])
        pos = torch.as_tensor(point_cloud[:, :3], dtype=torch.float).requires_grad_(False)
        reflectance = torch.as_tensor(point_cloud[:, self.reflectance_index], dtype=torch.float)

        local_shift = torch.mean(pos[:, :3], axis=0).requires_grad_(False)#torch.as_tensor([0., 0., 0.])
        pos = pos - local_shift
        scaling_factor = torch.sqrt((pos ** 2).sum(dim=1)).max()

        nan_mask = torch.isnan(pos).any(dim=1) | torch.isnan(reflectance) 
        pos = pos[~nan_mask]
        reflectance = reflectance[~nan_mask]

        if nan_mask.any(): print(f"Encountered NaN values in sample at index {index}")

        # combined_data = torch.cat((pos, reflectance.unsqueeze(1)), dim=1)  # Shape: (N, 4)
        # file_name = f"/home/harryowen/Desktop/voxels/point_cloud_{index}.txt"  # Unique file name for each point cloud
        # np.savetxt(file_name, combined_data.cpu().numpy(), fmt='%.6f', header='x y z reflectance')

        data = Data(pos=pos, reflectance=reflectance, local_shift=local_shift, sf = scaling_factor)

        return data        
        
from collections import OrderedDict
def load_model(path, model, device):
    checkpoint = torch.load(path, map_location=device)
    adjusted_state_dict = OrderedDict()
    for key, value in checkpoint['model_state_dict'].items():
        if key.startswith('module.'):
            key = key[7:]
        adjusted_state_dict[key] = value
    model.load_state_dict(adjusted_state_dict, strict=False)
    return model

@jit(nopython=True)
def percentile_90(arr):
    arr = arr[~np.isnan(arr)]
    sorted_arr = np.sort(arr)
    index = int(0.9 * len(sorted_arr))
    return sorted_arr[index]

@jit(nopython=True)
def percentile_75(arr):
    arr = arr[~np.isnan(arr)]
    sorted_arr = np.sort(arr)
    index = int(0.75 * len(sorted_arr))
    return sorted_arr[index]

@jit(nopython=True)
def percentile_50(arr):
    arr = arr[~np.isnan(arr)]
    sorted_arr = np.sort(arr)
    index = int(0.5 * len(sorted_arr))
    return sorted_arr[index]
    
@jit(nopython=True, parallel=True)
def compute_labels(nbr_classification, labels, is_wood, any_wood):
    num_neighborhoods = labels.shape[0]
    num_classes = nbr_classification.shape[1]
    for i in prange(num_neighborhoods):
        labels[i, 1] = np.mean(nbr_classification[i, :, -1])
        if any_wood != 1:
            over_threshold = nbr_classification[i, :, -2] > any_wood
            labels[i, 0] = np.where(np.any(over_threshold), 1, 0)  
        else:
            class_votes = np.zeros(num_classes)
            for j in range(num_classes):
                class_votes[j] = np.sum((nbr_classification[i, :, -2] == j) * nbr_classification[i, :, -1])
            labels[i, 0] = np.argmax(class_votes)  

    return labels

def collect_predictions(classification, original, args):
    original = original.drop(columns=[c for c in original.columns if c in ['label', 'pwood', 'pleaf']])
    indices_file = os.path.join('nbrs.npy')
    
    if os.path.exists(indices_file):
        indices = np.load(indices_file)
    else:
        kd_tree = KDTree(classification[:, :3])
        _, indices = kd_tree.query(original.values[:, :3], k = 16 if args.any_wood != 1 else 128)

    labels = np.zeros((original.shape[0], 2))
    labels = compute_labels(classification[indices], labels, args.is_wood, args.any_wood)
    original.loc[:, ['label', 'pwood']] = labels
    return original


#########################################################################################################
#                                       SEMANTIC INFERENCE FUNCTION                                     #
#                                       ==========================                                      #

def SemanticSegmentation(args):

    '''
    Setup Multi GPU processing. 
    '''

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    '''
    Setup model. 
    '''

    model = Net(num_classes=1).to(device)


    try:
        load_model(os.path.join(args.wdir,'model',args.model), model, device)
    except KeyError:
        raise Exception(f'No model loaded at {os.path.join(args.wdir,"model",args.model)}')

    #####################################################################################################
    
    '''
    Setup data loader. 
    '''

    test_dataset = TestingDataset(voxels=args.vxfile, device = device, max_pts=args.max_pts)
                                
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                              shuffle=False, drop_last=False, num_workers=0,
                              pin_memory=True)

    #####################################################################################################

    '''
    Initialise model
    '''

    model.eval()

    output_list = []

    with tqdm(total=len(test_loader), colour='white', ascii="▒█", bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}', desc = "Inference") as pbar:
        for _, data in enumerate(test_loader):

            data = data.to(device)
            
            with torch.cuda.amp.autocast():
                outputs = model(data)
                outputs = torch.nan_to_num(outputs)
            
            probs = torch.sigmoid(outputs).to(device)
            preds = (probs>=args.is_wood).type(torch.int64).cpu()
            preds = np.expand_dims(preds, axis=1)

            batches = np.unique(data.batch.cpu())
            pos = data.pos.cpu().numpy()
            probs_2d = np.expand_dims(probs.detach().cpu().numpy(), axis=1)  # Add an extra dimension to probs
            output = np.concatenate((pos, preds, probs_2d), axis=1)
            outputb = None

            for batch in batches:
                outputb = np.asarray(output[data.batch.cpu() == batch])
                #outputb[:, :3] = outputb[:, :3] * np.asarray(data.local_shift.cpu())[batch]
                outputb[:, :3] = outputb[:, :3] + np.asarray(data.local_shift.cpu())[3 * batch : 3 + (3 * batch)]
                output_list.append(outputb)
            pbar.update(1)
        
    classified_pc = np.vstack(output_list)
    #del outputb, outputs, batches, pos, output

    #####################################################################################################
    
    '''
    Choosing most confident labels using nearest neighbour search. 
    '''  

    if args.verbose: print("Spatially aggregating prediction probabilites and labels...")
    args.pc = collect_predictions(classified_pc, args.pc, args)

    '''
    Add ground classifications from CSF filter applied during preprocessing. 
    '''  

    if args.verbose: print("Normalising height using ground class...")

    if args.bare_ground:
        args.ground['label'] = 2
        args.pc = pd.concat([args.pc, args.ground], axis=0)
        args.pc.loc[args.grdidx, ['pwood']] = 0
        args.pc = NormaliseHeight(args.pc).height_normalise()
        #Remove ground points
        args.pc = args.pc[args.pc['label'] != 2]
    
    else:
        args.pc.loc[:, 'n_z']=args.pc.z

    '''
    Optional DBSCAN clean up. Run dbscan on args.pc points where label equals 1, make any labels where dbscan labels equal -1 to 0
    and make any labels associated with small clusters (less than 10 points) also 0. do nothing to pwood and pleaf
    '''

    if args.clean_up:
        if args.verbose: print("Running DBSCAN to remove small clusters...")
        dbscan_labels = fast_hdbscan.HDBSCAN(min_cluster_size=3).fit_predict(args.pc.loc[args.pc['label'] == 1, ['x', 'y', 'z','pwood']].values)
        args.pc.loc[args.pc['label'] == 1, ['label']] = np.where(dbscan_labels == -1, 0, 1)

    '''
    Save final classified point cloud. 
    '''

    headers = list(dict.fromkeys(args.headers+['n_z', 'label', 'pwood']))
    save_file(args.odir, args.pc.copy(), additional_fields= headers, verbose=False)    
    
    return args
