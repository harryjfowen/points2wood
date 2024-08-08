import datetime
start = datetime.datetime.now()
import resource
import os
import os.path as OP
import argparse
from src.preprocessing import *
from src.predicter import SemanticSegmentation
from tqdm import tqdm
import torch
import shutil
import sys
import numpy as np

'''
Minor functions-------------------------------------------------------------------------------------------------------------
'''

def get_path(location_in_points2wood=""):
    current_wdir = os.getcwd()
    
    # Find the last occurrence of "points2wood"
    last_index = current_wdir.rfind("points2wood") + len("points2wood")
    
    if last_index == -1 + len("points2wood"):  # "points2wood" not found
        raise ValueError('"points2wood" not found in the current working directory path')
    
    output_path = current_wdir[:last_index]
    
    if len(location_in_points2wood) > 0:
        output_path = os.path.join(output_path, location_in_points2wood)
    
    return output_path.replace("\\", "/")

def tidy_cols(args):
    
    args.point_cloud.rename(columns = {'scalar_refl':'refl'}, inplace = True)

    args.point_cloud.set_axis(labels=['label' if c.endswith('label') else c for c in args.point_cloud], axis=1, inplace=True)

    if 'label' in args.point_cloud.columns:
        args.point_cloud = args.point_cloud.astype({"label": int})


'''
-------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------
'''

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--point-cloud', '-p', default=[], nargs='+', type=str, help='list of point cloud files')
    parser.add_argument('--bare_ground', action='store_true', help='Whether point cloud has bare ground points to remove/classify')
    parser.add_argument('--reflectance_off', action='store_true', help="set all reflectance values to 0")
    
    parser.add_argument('--odir', type=str, default='.', help='output directory')
    
    parser.add_argument('--batch_size', default=8, type=int, help="If you get CUDA errors, try lowering this.")
    parser.add_argument('--num_procs', default=30, type=int, help="Number of CPU cores you want to use. If you run out of RAM, lower this.")

    parser.add_argument('--resolution', type=float, default=0.01, help='Resolution to which point cloud is downsampled [m]')
    parser.add_argument('--grid_size', type=float, nargs='+', default=[1.0, 2.0], help='Grid sizes for voxelization')

    parser.add_argument('--min_pts', type=int, default=8192, help='Minimum number of points in voxel')
    parser.add_argument('--max_pts', type=int, default=16384, help='Maximum number of points in voxel')

    parser.add_argument('--model', type=str, default='model.pth', help='path to candidate model')
    parser.add_argument('--is-wood', default=0.55, type=float, help='a probability above which points within KNN are classified as wood')
    parser.add_argument('--any-wood', default=1, type=float, help='a probability above which ANY point within KNN is classified as wood')

    parser.add_argument('--clean_up', action='store_true', help="Apply dbscan to wood labels to tidy up output [noise and small clusters]")
                    
    parser.add_argument('--output_fmt', default='ply', help="file type of output")
    parser.add_argument('--verbose', action='store_true', help="print stuff")

    args = parser.parse_args()

    if args.verbose:
        print('\n---- parameters used ----')
        for k, v in args.__dict__.items():
            if k == 'pc': v = '{} points'.format(len(v))
            if k == 'global_shift': v = v.values
            print('{:<35}{}'.format(k, v)) 

    args.wdir = get_path()
    args.mode = 'predict' if 'predict' in sys.argv[0] else 'train'
    args.reflectance = False


    '''
    Sanity check---------------------------------------------------------------------------------------------------------
    '''
    if args.point_cloud == '':
        raise Exception('no input specified, please specify --point-cloud')
    
    # Check if all files in the list exist
    for point_cloud_file in args.point_cloud:
        if not os.path.isfile(point_cloud_file):
            raise FileNotFoundError(f'Point cloud file not found: {point_cloud_file}')

        '''
    If voxel file on disc, delete it.
    '''    
    
    #Create output directories
    path = OP.dirname(args.point_cloud[0])
    args.vxfile = OP.join(path, "voxels")

    if os.path.exists(args.vxfile): shutil.rmtree(args.vxfile)

    for point_cloud_file in args.point_cloud:

        '''
        Handle input and output file paths-----------------------------------------------------------------------------------
        '''
        
        path = OP.dirname(point_cloud_file)
        file = OP.splitext(OP.basename(point_cloud_file))[0] + "-lw.ply"
        args.odir = OP.join(path, file)

        '''
        Preprocess data into voxels------------------------------------------------------------------------------------------
        '''

        if args.verbose: print('\n----- Preprocessing started -----')

        args.pc, args.headers = load_file(filename=point_cloud_file, additional_headers=True, verbose=False)
        args.pc = args.pc.drop(columns=['n_z', 'label', 'pwood', 'pleaf'], errors='ignore')

        args.pc.rename(columns=lambda x: x.replace('scalar_', '') if 'scalar_' in x else x, inplace=True)
        args.pc.rename(columns={'refl': 'reflectance'}, inplace=True)

        args.headers = [header for header in args.pc.columns[3:] if header not in ['n_z', 'label', 'pWood', 'pLeaf']]

        if 'reflectance' in args.pc.columns or args.pc.columns.str.contains('refl').any():
            args.reflectance = True
            print('Reflectance detected')
        else:
            zeros_reflectance = np.zeros(len(args.pc))
            #random_reflectance = (torch.rand(args.pc.size(0), 1) - 0.5) * 2 * 1e-6
            args.pc.insert(3, 'reflectance', zeros_reflectance)
            print('No reflectance detected, column added with zeros.')
                    
        if args.reflectance_off:
            if 'reflectance' in args.pc.columns:
                args.pc.drop(columns=['reflectance'], inplace=True)
                zeros_reflectance = np.zeros(len(args.pc))
                args.pc.insert(3, 'reflectance', zeros_reflectance)
            else:
                zeros_reflectance = np.zeros(len(args.pc))
                args.pc.insert(3, 'reflectance', zeros_reflectance)
                print('No reflectance, column added with random values from -1 to 1 and computing prior')
        
        os.makedirs(args.vxfile, exist_ok=True)

        print(f'Voxelising to {args.grid_size} grid sizes')
        preprocess(args)
        
        if args.verbose:
            print(f'peak memory: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6}')
            print(f'runtime: {(datetime.datetime.now() - start).seconds}')
        
        '''
        Run semantic training------------------------------------------------------------------------------------------------
        '''
        if args.verbose: print('\n----- Semantic segmenation started -----')
        
        SemanticSegmentation(args)
        torch.cuda.empty_cache()

        if os.path.exists(args.vxfile):
            shutil.rmtree(args.vxfile)

        if args.verbose:
            print(f'peak memory: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6}')
            print(f'runtime: {(datetime.datetime.now() - start).seconds}')
