import datetime
start = datetime.datetime.now()
import resource

import argparse, glob, os
import numpy as np
import shutil
from src.trainer import SemanticTraining
from src.preprocessing import *
from src.io import load_file
import shutil 
import sys

'''
Minor functions-------------------------------------------------------------------------------------------------------------
'''

def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

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

'''
---------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------
'''

if __name__ == '__main__':

        parser = argparse.ArgumentParser()

        parser.add_argument('--device', type=str, default='cuda', help='Insert either "cuda" or "cpu"')
        parser.add_argument('--num_procs', type=int, default=1, help='Number of cpu cores to use')

        parser.add_argument('--num_epochs', default=2, type=int, metavar='N', help='number of total epochs to run')
        parser.add_argument('--checkpoint_saves', default=1, type=int, metavar='N', help='number of times to save model')
        parser.add_argument('--model', type=str, default='model.pth', help='Name of global model [e.g. model.pth]')
        parser.add_argument('--in_memory', action='store_true', help='Load all data into memory')
                
        parser.add_argument('--resolution', type=float, default=0.01, help='Resolution to which point cloud is downsampled [m]')
        parser.add_argument('--grid_size', type=float, nargs='+', default=[1.0, 2.0], help='Grid sizes for voxelization')

        parser.add_argument('--min_pts', type=int, default=8192, help='Minimum number of points in voxel')
        parser.add_argument('--max_pts', type=int, default=16384, help='Maximum number of points in voxel')
        parser.add_argument('--batch_size', type=int, default=2, help='Batch size for cuda processing [Lower less memory usage]')
        parser.add_argument('--augmentation', action='store_true', help="Perform data augmentation")
        parser.add_argument('--preprocess', action='store_true', help="Preprocess point clouds into voxels")
        parser.add_argument('--test', action='store_true', help="Perform model testing during training")
        parser.add_argument('--bare_ground', action='store_true', help="Point cloud contains ground points")
        parser.add_argument('--tune', action='store_true', help="Tune model hyperparameters with lower learning rate schedule")
        parser.add_argument('--stop_early', action='store_true', help="Break training run if testing loss continually increases")
        parser.add_argument('--wandb', action='store_true', help="Use wandb for logging")
        parser.add_argument('--verbose', action='store_true', help="print stuff")

        args = parser.parse_args()
        
        args.wdir = get_path()
        args.reflectance_off = False

        args.mode = 'predict' if 'predict' in sys.argv[0] else 'train'
        
        print('Mode: {}'.format(args.mode))

        '''
        Organise model checkpointing-------------------------------------------------------------------------------------------
        '''

        args.checkpoints = np.arange(0, args.num_epochs+1, int(args.num_epochs / args.checkpoint_saves))

        old_checkpoints = glob.glob(os.path.join(args.wdir,'checkpoints/*.pth'))
        if len(old_checkpoints) > 0:
                shutil.make_archive(os.path.join(args.wdir,'checkpoints_backup'), 'zip', os.path.join(args.wdir,'checkpoints'))
        for f in old_checkpoints:
                os.remove(f)
        
        '''
        Preprocess data into voxels and write to disk----------------------------------------------------------------------------
        '''

        args.trfile = os.path.join(args.wdir, "data", "train", "voxels")
        args.tefile = os.path.join(args.wdir, "data", "test", "voxels")
        args.vxfile = args.trfile

        if args.preprocess:

                if os.path.exists(args.trfile): shutil.rmtree(args.trfile)

                if args.verbose: print('\n----- Preprocessing started -----')

                args.point_clouds = glob.glob(os.path.join(args.wdir + '/data/*/*.ply'))                

                for i, p in enumerate([word for word in args.point_clouds if 'train' in word]):

                        args.pc, args.headers = load_file(filename=p, additional_headers=True, verbose=True)
                        args.pc.rename(columns=lambda x: x.replace('scalar_', '') if 'scalar_' in x else x, inplace=True)
                        args.pc.rename(columns={'refl': 'reflectance'}, inplace=True)
                        args.pc.rename(columns={'truth': 'label'}, inplace=True)
                        args.headers = list(args.pc.columns[3:])
                        if 'reflectance' in args.pc.columns or args.pc.columns.str.contains('refl').any():
                              args.reflectance = True
                              print('Reflectance detected')
                        else: 
                                args.prior = True
                                zeros_reflectance = np.zeros(len(args.pc))
                                args.pc.insert(3, 'reflectance', zeros_reflectance)
                                print('No reflectance detected, column added with zeros.')
                                
                        os.makedirs(args.trfile, exist_ok=True)
                        preprocess(args)

                if args.test:

                        if os.path.exists(args.tefile): shutil.rmtree(args.tefile)
                        
                        print("\nTesting")

                        args.mode = 'test'

                        for i, p in enumerate([word for word in args.point_clouds if 'test' in word]):

                                args.pc, args.headers = load_file(filename=p, additional_headers=True, verbose=True)
                                args.pc.rename(columns=lambda x: x.replace('scalar_', '') if 'scalar_' in x else x, inplace=True)
                                args.pc.rename(columns={'refl': 'reflectance'}, inplace=True)
                                args.pc.rename(columns={'truth': 'label'}, inplace=True)
                                args.vxfile = args.tefile
                                if 'reflectance' in args.pc.columns or args.pc.columns.str.contains('refl').any():
                                       args.reflectance = True
                                else:
                                       args.prior = True
                                       zeros_reflectance = np.zeros(len(args.pc))
                                       args.pc.insert(3, 'reflectance', zeros_reflectance)
                                       print('No reflectance detected, column added with zeros.')
                                
                                os.makedirs(args.tefile, exist_ok=True)
                                preprocess(args)

                if args.verbose:
                        print(f'peak memory: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6}')
                        print(f'runtime: {(datetime.datetime.now() - start).seconds}')

        if args.augmentation:
                print('Training with data augmentation performed on 25% of samples')


        '''
        Sanity checks-------------------------------------------------------------------------------------------------------------
        '''

        if len(args.checkpoints) == 0:
                args.checkpoints == np.asarray([args.num_epochs-1])


        '''
        Run semantic training-----------------------------------------------------------------------------------------------------
        '''
        if args.verbose: print('\n----- Semantic segmenation started -----')

        SemanticTraining(args)

        if args.verbose:
                print(f'peak memory: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6}')
                print(f'runtime: {(datetime.datetime.now() - start).seconds}')


