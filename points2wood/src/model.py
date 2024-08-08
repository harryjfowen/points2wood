import torch
import torch.nn.functional as F
from torch_geometric.nn import knn_interpolate
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, GroupNorm as GN, BatchNorm1d as BN
from torch_geometric.nn import PointNetConv, radius, voxel_grid, knn, global_max_pool
from src.pointnet import PointNetConv
from torch_geometric.nn.pool.consecutive import consecutive_cluster
import math

def select_num_groups(channels, preferred_group_size=8):
    for group_size in range(preferred_group_size, 0, -1):
        if channels % group_size == 0:
            return channels // group_size
    return 1

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (torch.nn.Conv1d, torch.nn.Linear)):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
            if isinstance(m, torch.nn.Conv1d):
                torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')

class DepthwiseSeparableConv1d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super(DepthwiseSeparableConv1d, self).__init__()
        
        self.depthwise_conv = torch.nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels  # Groups parameter set to in_channels for depthwise convolution
        )
        
        #self.depthwise_ln = GN(select_num_groups(in_channels), in_channels)
        self.depthwise_bn = torch.nn.BatchNorm1d(in_channels)
        
        self.pointwise_conv = torch.nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=1  # Pointwise convolution uses 1x1 kernel size
        )
        
        #self.pointwise_ln = GN(select_num_groups(in_channels), out_channels)
        self.pointwise_bn = torch.nn.BatchNorm1d(in_channels)
        
    def forward(self, x):

        out = self.depthwise_conv(x)
        out = self.depthwise_bn(out)
        out = torch.nn.functional.relu(out, inplace=True)
        
        out = self.pointwise_conv(out)
        out = self.pointwise_bn(out)
        out = torch.nn.functional.relu(out, inplace=True)

        return out
    
class InvertedResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor=4):
        super(InvertedResidualBlock, self).__init__()
        self.expansion_factor = expansion_factor
        expanded_channels = in_channels * expansion_factor

        self.expand = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels, expanded_channels, kernel_size=1),
            #GN(select_num_groups(expanded_channels), expanded_channels),
            torch.nn.BatchNorm1d(expanded_channels),
            torch.nn.ReLU(),
        )

        self.conv = torch.nn.Sequential(
            DepthwiseSeparableConv1d(expanded_channels, expanded_channels, kernel_size=1),
            #GN(select_num_groups(expanded_channels), expanded_channels),
            torch.nn.BatchNorm1d(expanded_channels),
            torch.nn.ReLU(),
            DepthwiseSeparableConv1d(expanded_channels, expanded_channels, kernel_size=1),  # Adjusted output channels here
            #GN(select_num_groups(expanded_channels), expanded_channels)
            torch.nn.BatchNorm1d(expanded_channels),
        )

        self.project = torch.nn.Sequential(
            torch.nn.Conv1d(expanded_channels, out_channels, kernel_size=1),
            torch.nn.BatchNorm1d(out_channels)
            #GN(select_num_groups(out_channels), out_channels)
        )

        if in_channels != out_channels:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv1d(in_channels, out_channels, kernel_size=1),
                #GN(select_num_groups(out_channels), out_channels)
                torch.nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = torch.nn.Sequential()

    def forward(self, x):
        
        residual = x

        out = x.unsqueeze(0).permute(0, 2, 1)
        
        out = self.expand(out)
        out = self.conv(out)
        out = self.project(out)

        out = out.permute(0, 2, 1).squeeze(0)
        
        residual = self.shortcut(residual)
        out += residual
        out = torch.nn.functional.relu(out, inplace = True)

        return out

class SAModule(torch.nn.Module):
    def __init__(self, resolution, radius, k, NN, RNN):
        super(SAModule, self).__init__()
        self.resolution = resolution
        self.radius = radius
        self.k = k
        self.conv = PointNetConv(local_nn=NN, global_nn=None, add_self_loops=False, radius=radius)
        self.residual_block = InvertedResidualBlock(RNN, RNN)
        self.reflectance_weight = torch.nn.Parameter(torch.ones(1))

    def random_sample(self, num_points):
        num_samples = int(num_points * 0.5)
        idx = torch.randperm(num_points)[:num_samples]
        idx, _ = torch.sort(idx)
        return idx

    def voxelsample(self, pos, batch, resolution):
        voxel_indices = voxel_grid(pos, resolution, batch)
        _, idx = consecutive_cluster(voxel_indices)
        return idx
    
    def forward(self, x, pos, batch, reflectance, sf):

        if torch.sum(reflectance) != 0:
            w = reflectance * self.reflectance_weight
            pos = torch.cat([pos[:, :3], w.unsqueeze(-1)], dim=-1)

        if self.training: 
            idx = self.random_sample(pos.shape[0])
        else:
            idx = self.voxelsample(pos[:, :3], batch, self.resolution)
        
        if self.resolution == 0.04:
            row, col = radius(pos[:, :3], pos[idx, :3], self.resolution*2, batch, batch[idx], max_num_neighbors=self.k)
        else: 
            row, col = knn(x=pos[:, :3], y=pos[idx, :3], k=self.k, batch_x=batch, batch_y=batch[idx])
        
        edge_index = torch.stack([col, row], dim=0)

        pos[:, :3] = pos[:, :3] / sf[batch].unsqueeze(-1)
        x = self.conv(x, (pos, pos[idx]), edge_index)
        pos[:, :3] = pos[:, :3] * sf[batch].unsqueeze(-1)

        x = self.residual_block(x)

        pos, batch, reflectance = pos[idx, :3], batch[idx], reflectance[idx]
        return x, pos, batch, reflectance, sf
    
class GlobalSAModule(torch.nn.Module):
    def __init__(self, NN, ANN):
        super(GlobalSAModule, self).__init__()
        self.NN = NN

    def forward(self, x, pos, batch, reflectance, sf):
        #pos = torch.cat([pos[:, :3], reflectance.unsqueeze(-1)], dim=-1)
        x = self.NN(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        reflectance = reflectance.new_zeros(x.size(0))
        return x, pos, batch, reflectance, sf
    
def MLP(channels):
    return Seq(*[
        Seq(*( [Lin(channels[i - 1], channels[i]), ReLU()] + ([BN(channels[i])] if i != 1 else []) ))
        for i in range(1, len(channels))
    ])

class FPModule(torch.nn.Module):
    def __init__(self, k, NN):
        super(FPModule, self).__init__()
        self.k = k
        self.NN = NN

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.NN(x)
        return x, pos_skip, batch_skip


class Net(torch.nn.Module):
    def __init__(self, num_classes, C=32):
        super(Net, self).__init__()

        self.stem_mlp = MLP([3, C])

        self.sa1_module = SAModule(0.04, 0.04, 32, MLP([C + 4, C * 2, C * 4]), C * 4)
        self.sa2_module = SAModule(0.08, 0.08, 32, MLP([C * 4 + 4, C * 6, C * 8]), C * 8)
        self.sa3_module = SAModule(0.16, 0.16, 32, MLP([C * 8 + 4, C * 12, C * 16]), C * 16)
        self.sa4_module = GlobalSAModule(MLP([C * 16 + 3, C * 16, C * 16]), C * 16)

        self.fp4_module = FPModule(2, MLP([C * 32, C * 24, C * 16]))
        self.fp3_module = FPModule(2, MLP([C * 24, C * 20, C * 16]))
        self.fp2_module = FPModule(2, MLP([C * 20, C * 16, C * 16]))
        self.fp1_module = FPModule(2, MLP([C * 17, C * 16, C * 16]))

        self.conv1 = torch.nn.Conv1d(C * 16, C * 16, 1)
        self.conv2 = torch.nn.Conv1d(C * 16, num_classes, 1)
        #self.norm = GN(select_num_groups(C*16), C * 16)
        self.norm = torch.nn.BatchNorm1d(C * 16)

        initialize_weights(self)

    def forward(self, data):

        #data.x = self.stem_mlp(torch.cat([data.pos[:,:3], data.reflectance.unsqueeze(-1)], dim=-1))
        data.x = self.stem_mlp(data.pos[:,:3])
        sa0_out = (data.x, data.pos, data.batch, data.reflectance, data.sf)

        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        sa4_out = self.sa4_module(*sa3_out)

        fp4_out = self.fp4_module(*sa4_out[:-2], *sa3_out[:-2])
        fp3_out = self.fp3_module(*fp4_out, *sa2_out[:-2])
        fp2_out = self.fp2_module(*fp3_out, *sa1_out[:-2])
        x, _, _ = self.fp1_module(*fp2_out, *sa0_out[:-2])

        x = self.conv1(x.unsqueeze(dim=0).permute(0, 2, 1))
        x = F.relu(self.norm(x), inplace=True)
        x = torch.squeeze(self.conv2(x)).to(torch.float)

        return x

