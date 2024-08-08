#import cupy as cp
# mempool = cp.get_default_memory_pool()
# pinned_mempool = cp.get_default_pinned_memory_pool()
import numpy as np
from concurrent.futures import ThreadPoolExecutor

##---------------------------------------------------------------------------------------------------------
## GPU
##---------------------------------------------------------------------------------------------------------

# class PriorGPU():
#     def __init__(self, pc, indices):
#         self.arr = pc[['x', 'y', 'z']].values
#         self.indices = indices
#         self.k = indices.shape[1]
#         #self.available_mem = cp.cuda.runtime.getDeviceProperties(0)['totalGlobalMem']
#         self.available_mem, _ = cp.cuda.runtime.memGetInfo()
#         self.required_mem = np.ceil(self.arr.nbytes * 3.5 + self.indices.nbytes)
#         self.block_size = self.available_mem/self.required_mem
#         self.block_size = int(self.arr.shape[0]*self.block_size)
#         num_blocks = int(np.ceil(self.arr.shape[0] / self.block_size))
#         self.blocks = [np.arange(i*self.block_size, min((i+1)*self.block_size, self.arr.shape[0])) for i in range(num_blocks)]
#         self.results_blocks = np.zeros(self.arr.shape[0], dtype=np.float16)

#     def compute(self):
#         for b, _ in enumerate(self.blocks):

#             if len(self.blocks)==1:
#                 array_gpu = cp.asarray(self.arr[self.indices])
#             else: 
#                 array_gpu = cp.asarray(self.arr[self.indices[self.blocks[b]]])
#             array_gpu[:, :self.k, :] -= cp.mean(array_gpu[:, :self.k, :], axis=1, keepdims=True)
#             cov = cp.einsum('ijk,ijl->ikl', array_gpu[:, :self.k, :], array_gpu[:, :self.k, :]) / self.k 
#             del array_gpu; mempool.free_all_blocks() 
#             evals = cp.linalg.svd(cov, compute_uv=False)
#             del cov; mempool.free_all_blocks() 
#             ratios = (evals.T / cp.sum(evals, axis=1)).T
#             del evals; mempool.free_all_blocks() 
#             ratios[cp.isnan(ratios)] = 0
#             prior = (ratios[:, 0] - ratios[:, 2]) / ratios[:, 0]
#             del ratios; mempool.free_all_blocks() 
#             self.results_blocks[self.blocks[b]] = cp.asnumpy(prior)
#             del prior; mempool.free_all_blocks()
#             pinned_mempool.free_all_blocks()
        
#         array_gpu = None
#         cp.cuda.Device().synchronize()
#         cp.get_default_memory_pool().free_all_blocks()

#         return self.results_blocks

##---------------------------------------------------------------------------------------------------------
## CPU
##---------------------------------------------------------------------------------------------------------

class PriorCPU():
    def __init__(self, pc, indices, k=128):
        self.arr = pc[['x', 'y', 'z']].values
        self.indices = indices
        self.k = k
        self.results_blocks = np.zeros(self.arr.shape[0], dtype=np.float16)
        self.block_size = 100000  # Adjust this value to optimize performance
        num_blocks = int(np.ceil(self.arr.shape[0] / self.block_size))
        self.blocks = [np.arange(i*self.block_size, min((i+1)*self.block_size, self.arr.shape[0])) for i in range(num_blocks)]

    def prior(self, block):
        array_cpu = self.arr[self.indices[block, :self.k]]
        array_cpu[:, :self.k, :] -= np.mean(array_cpu[:, :self.k, :], axis=1, keepdims=True)
        cov = np.einsum('ijk,ijl->ikl', array_cpu[:, :self.k, :], array_cpu[:, :self.k, :]) / self.k 
        evals = np.linalg.svd(cov, compute_uv=False)
        ratios = (evals.T / np.sum(evals, axis=1)).T
        ratios[np.isnan(ratios)] = 0
        linearity = (ratios[:, 0] - ratios[:, 1]) / ratios[:, 0]
        linearity[np.isnan(linearity) | (linearity == 0)] = 0.5
        planarity = (ratios[:, 1] - ratios[:, 2]) / ratios[:, 0]
        planarity[np.isnan(planarity) | (planarity == 0)] = 0.5
        max_prior = np.maximum(linearity, planarity)
        return block, max_prior
    
    def compute(self):
        with ThreadPoolExecutor(max_workers=32) as executor:
            results = list(executor.map(self.prior, self.blocks))
        for block, prior in results:
            self.results_blocks[block] = prior
        return self.results_blocks
    
##---------------------------------------------------------------------------------------------------------

# class PriorCPU():
#     def __init__(self, pc, indices):
#         self.arr = pc[['x', 'y', 'z']].values
#         self.indices = indices
#         self.k = indices.shape[1]
#         self.results_blocks = np.zeros(self.arr.shape[0], dtype=np.float16)
#         self.block_size = 10000  # Adjust this value to optimize performance
#         num_blocks = int(np.ceil(self.arr.shape[0] / self.block_size))
#         self.blocks = [np.arange(i*self.block_size, min((i+1)*self.block_size, self.arr.shape[0])) for i in range(num_blocks)]

#     def prior(self, block):
#         max_prior = np.zeros((block.shape[0],), dtype=np.float16)
#         for i, k in enumerate([self.k // 2, self.k]):
#             neighbors = self.arr[self.indices[block, :k]]
#             neighbors -= np.mean(neighbors, axis=1, keepdims=True)
#             cov = np.einsum('ijk,ijl->ikl', neighbors, neighbors) / k 
#             evals = np.linalg.svd(cov, compute_uv=False)
#             ratios = (evals.T / np.sum(evals, axis=1)).T
#             ratios[np.isnan(ratios)] = 0
#             if i == 0:  # For the lower k, compute linearity
#                 linearity = (ratios[:, 0] - ratios[:, 1]) / ratios[:, 0]
#                 linearity[np.isnan(linearity) | (linearity == 0)] = 0.5
#                 max_prior = linearity
#             else:  # For the higher k, compute planarity
#                 planarity = (ratios[:, 1] - ratios[:, 2]) / ratios[:, 0]
#                 planarity[np.isnan(planarity) | (planarity == 0)] = 0.5
#                 max_prior = np.maximum(max_prior, planarity)
#         return block, max_prior

#     def compute(self):
#         with ThreadPoolExecutor(max_workers=32) as executor:
#             results = list(executor.map(self.prior, self.blocks))
#         for block, prior in results:
#             self.results_blocks[block] = prior
#         return self.results_blocks
    
# class PriorCPU():
#     def __init__(self, pc, indices):
#         self.arr = pc
#         self.indices = indices
#         self.k = indices.shape[1]
#         self.results_blocks = np.zeros(self.arr.shape[0], dtype=np.float16)
#         self.block_size = 10000  # Adjust this value to optimize performance
#         num_blocks = int(np.ceil(self.arr.shape[0] / self.block_size))
#         self.blocks = [np.arange(i*self.block_size, min((i+1)*self.block_size, self.arr.shape[0])-1) for i in range(num_blocks)]
#     def prior(self, block, k=[64, 128]):
#         block = block[block < self.indices.shape[0]]
#         max_prior = np.zeros((block.shape[0],), dtype=np.float16)
#         for i, neighbors_count in enumerate(k):
#             neighbors = self.arr[self.indices[block, :neighbors_count]]
#             neighbors -= np.mean(neighbors, axis=1, keepdims=True)
#             cov = np.einsum('ijk,ijl->ikl', neighbors, neighbors) / neighbors_count
#             evals = np.linalg.svd(cov, compute_uv=False)
#             ratios = (evals.T / np.sum(evals, axis=1)).T
#             ratios[np.isnan(ratios)] = 0
#             if i == 0:  # For the lower k, compute linearity
#                 prior = (ratios[:, 0] - ratios[:, 1]) / ratios[:, 0]
#             else:  # For the higher k, compute planarity
#                 prior = (ratios[:, 1] - ratios[:, 2]) / ratios[:, 0]
#             prior[np.isnan(prior) | (prior == 0)] = 0.5
#             max_prior = np.maximum(max_prior, prior)
#         return block, max_prior
#     def compute(self):
#         with ThreadPoolExecutor(max_workers=32) as executor:
#             results = list(executor.map(self.prior, self.blocks))
#         for block, prior in results:
#             self.results_blocks[block] = prior
#         return self.results_blocks