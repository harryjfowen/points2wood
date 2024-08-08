__author__ = "Matheus Boni Vicari"
__copyright__ = "Copyright 2018-2019"
__credits__ = ["Matheus Boni Vicari"]
__license__ = "GPL3"
__version__ = "1.0.1"
__maintainer__ = "Matheus Boni Vicari"
__email__ = "matheus.boni.vicari@gmail.com"
__status__ = "Development"

import os
import sys
import numpy as np
from shortest_path import (array_to_graph, extract_path_info)
from downsample import downsample_cloud, upsample_cloud
from inout import load_file, save_file
import pandas as pd


if __name__ == '__main__':

    # Declaring list of files to process.
    # Get the string from command line arguments
    files_string = sys.argv[1]

    # Split the string into a list of file paths
    filelist = files_string.split()

    # Declaring downsample size of 10 cm.
    downsample_size = 0.05

    for f in filelist:

        # Detecting file name.
        fname = os.path.splitext(os.path.basename(f))[0]

        print('Processing %s' % fname)

        # Loading point cloud and selecting only xyz coordinates.
        point_cloud, headers = load_file(filename=f, additional_headers=True, verbose=False)

        # Check if 'label' is in headers
        xyz = point_cloud[['x', 'y', 'z']].values

        # Downsample point cloud to speed up processing.
        downsample_indices, downsample_nn = downsample_cloud(xyz,
                                                             downsample_size,
                                                             True, True)
        downsample_pc = xyz[downsample_indices, :3]

        # Growth factor. Each point adds 3 new points to graph.
        kpairs = 3

        # NN search of the whole point cloud. This allocates knn indices
        # for each point in order to grow the graph. The more segmented (gaps)
        # the cloud is, the larger knn has to be.
        knn = 100

        # Maximum distance between points. If distance > threshold, neighboring
        # point is not added to graph.
        nbrs_threshold = 0.15

        # When initial growth process is broken (gap in the cloud) and no
        # other point can be added, incease threshold to include missing points.
        nbrs_threshold_step = 0.05

        # Base/root point of the point cloud.
        base_point = np.argmin(downsample_pc[:, 2])

        # Generates graph from numpy array.
        G = array_to_graph(downsample_pc, base_point, kpairs, knn, nbrs_threshold,
                           nbrs_threshold_step)

        # Extracts shortest path info from G and generate nodes point cloud.
        nodes_ids, distance, path_list = extract_path_info(G, base_point,
                                                           return_path=True)

        # Upscaling the point cloud and distance values.
        nodes_ids = np.array(nodes_ids)
        # Get the upscaled set of indices for the final points.
        upscale_ids = upsample_cloud(downsample_indices[nodes_ids],
                                     downsample_nn)
        # Allocating upscale_distance. Looping over the original downsample
        # indices and distance values. This will be used to retrieve each
        # downsampled points' neighbors indices and apply the distance value
        # to them.
        upscale_distance = np.full(upscale_ids.shape[0], np.nan)
        for n, d in zip(downsample_indices[nodes_ids], distance):
            up_ids = downsample_nn[n]
            upscale_distance[up_ids] = d

        # Generating the upscaled cloud [use original cloud with leaf sep column]
        upscale_cloud = point_cloud[upscale_ids]

        # Calculating difference array and preparing output point cloud.
        scaled_height = upscale_cloud[:, 2] - np.min(upscale_cloud[:, 2])
        diff = np.abs(scaled_height - upscale_distance)
	

        # Create a DataFrame from the upscale_cloud array
        out_cloud = pd.DataFrame(upscale_cloud, columns=headers)

        # Add the 'upscale_distance' column to the DataFrame
        out_cloud['pathlength'] = upscale_distance

        # CREATE OUTPUT DIRECTORY
        out_dir = os.path.dirname(os.path.dirname(f)) + "/trees-pl/"
        if not os.path.exists(out_dir):
          os.mkdir(out_dir)

        save_file(out_cloud, out_dir, additional_fields= headers + ['pathlength'], verbose=False)
