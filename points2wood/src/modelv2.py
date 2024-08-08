import torch
import torch.nn.functional as F
from torch_geometric.nn import knn_interpolate
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Parameter, Softmax, BatchNorm1d as BN, GroupNorm as GN
from torch_geometric.nn import PointNetConv, fps, knn, radius, voxel_grid, global_max_pool
from torch_scatter import scatter_max, scatter_mean, scatter_std
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from src.pointnetv2 import PointNetConv

def select_num_groups(channels, preferred_group_size=8):
    for group_size in range(preferred_group_size, 0, -1):
        if channels % group_size == 0:
            return channels // group_size
    return 1
        
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
        
        #self.depthwise_bn = torch.nn.BatchNorm1d(in_channels)
        self.depthwise_gn = GN(select_num_groups(in_channels), in_channels)
        
        self.pointwise_conv = torch.nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=1  # Pointwise convolution uses 1x1 kernel size
        )
        
        #self.pointwise_bn = torch.nn.BatchNorm1d(out_channels)
        self.pointwise_gn = GN(select_num_groups(out_channels), out_channels)
        
    def forward(self, x):
        out = self.depthwise_conv(x)
        out = self.depthwise_gn(out)
        out = torch.nn.functional.relu(out, inplace=True)
        
        out = self.pointwise_conv(out)
        out = self.pointwise_gn(out)
        out = torch.nn.functional.relu(out, inplace=True)
        
        return out
    
class InvertedResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor=4):
        super(InvertedResidualBlock, self).__init__()
        self.expansion_factor = expansion_factor
        expanded_channels = in_channels * expansion_factor

        self.expand = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels, expanded_channels, kernel_size=1),
            #torch.nn.BatchNorm1d(expanded_channels),
            GN(select_num_groups(in_channels), expanded_channels),
            torch.nn.ReLU(),
        )

        self.conv = torch.nn.Sequential(
            DepthwiseSeparableConv1d(expanded_channels, expanded_channels, kernel_size=1),
            #torch.nn.BatchNorm1d(expanded_channels),
            GN(select_num_groups(expanded_channels), expanded_channels),
            torch.nn.ReLU(),
            DepthwiseSeparableConv1d(expanded_channels, expanded_channels, kernel_size=1),  # Adjusted output channels here
            #torch.nn.BatchNorm1d(expanded_channels)
            GN(select_num_groups(expanded_channels), expanded_channels)
        )

        self.project = torch.nn.Sequential(
            torch.nn.Conv1d(expanded_channels, out_channels, kernel_size=1),
            #torch.nn.BatchNorm1d(out_channels)
            GN(select_num_groups(out_channels), out_channels)

        )

        if in_channels != out_channels:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv1d(in_channels, out_channels, kernel_size=1),
                #torch.nn.BatchNorm1d(out_channels)
                GN(select_num_groups(out_channels), out_channels)
            )
        else:
            self.shortcut = torch.nn.Sequential()

    def forward(self, x):
        x = x.unsqueeze(0)
        x = x.permute(0, 2, 1)
        residual = x

        out = self.expand(x)
        out = self.conv(out)
        out = self.project(out)

        residual = self.shortcut(residual)
        out += residual
        out = torch.nn.functional.relu(out, inplace = False)

        out = out.squeeze(0)
        out = out.permute(1, 0)
        return out

class SAModule(torch.nn.Module):
    def __init__(self, resolution, radius, k, NN):
        super(SAModule, self).__init__()
        self.ratio = 0.5
        self.resolution = resolution
        self.radius = radius
        self.k = k
        self.conv = PointNetConv(local_nn=MLP(NN), global_nn=None, add_self_loops=False, radius=radius)
        self.residual_block = InvertedResidualBlock(NN[-1], NN[-1])
        self.reflectance_mlp = MLP([2, NN[-1]])
        self.gating_mlp = torch.nn.Sequential(torch.nn.Linear(NN[-1], NN[-1]),torch.nn.Sigmoid())
        self.merging_mlp = MLP([int(NN[-1]*2), NN[-1]])
    
    def voxelsample(self, pos, batch, resolution):
        voxel_indices = voxel_grid(pos, resolution, batch)
        _, idx = consecutive_cluster(voxel_indices)
        return idx
    
    def random_sample(self, num_points):
        num_samples = int(num_points * 0.5)
        idx = torch.randperm(num_points)[:num_samples]
        idx, _ = torch.sort(idx)
        return idx
    
    def forward(self, x, pos, reflectance, batch, sf):

        idx = self.voxelsample(pos, batch, self.resolution)
        #idx = self.random_sample(pos.shape[0])

        #row, col = radius(pos[:, :3], pos[idx, :3], self.radius, batch, batch[idx], max_num_neighbors=self.k)
        row, col = knn(x=pos[:, :3], y=pos[idx, :3], k=self.k, batch_x=batch, batch_y=batch[idx])

        edge_index = torch.stack([col, row], dim=0)
        
        pos = pos / sf[batch].unsqueeze(-1)
        x = self.conv(x, (pos, pos[idx]), edge_index)
        pos = pos * sf[batch].unsqueeze(-1)

        if reflectance is not None and torch.any(reflectance[col] != 0):
            reflectance_mean = scatter_mean(reflectance[col], row, dim=0).unsqueeze(1)
            reflectance_std = scatter_std(reflectance[col], row, dim=0).unsqueeze(1)
            reflectance_features = torch.cat([reflectance_mean, reflectance_std], dim=1)
            reflectance_features = self.reflectance_mlp(reflectance_features)
            
            gate = self.gating_mlp(reflectance_features)
            reflectance_features = gate * reflectance_features

            x = torch.cat([x, reflectance_features], dim=1)
            x = self.merging_mlp(x)

        x = self.residual_block(x)
        pos, batch, reflectance = pos[idx, :3], batch[idx], reflectance[idx]

        return x, pos, reflectance, batch, sf
    
def MLP(channels):
    return Seq(*[
        Seq(*( [Lin(channels[i - 1], channels[i]), ReLU()] + ([GN(select_num_groups(channels[i]),channels[i])] if i != 1 else []) ))
        for i in range(1, len(channels))
    ])

class FPModule(torch.nn.Module):
    def __init__(self, k, NN):
        super(FPModule, self).__init__()
        self.k = k
        self.NN = MLP(NN)

    def forward(self, x, pos, reflectance, batch, x_skip, pos_skip, reflectance_skip, batch_skip):
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.NN(x)
        return x, pos_skip, reflectance_skip, batch_skip
    

class Net(torch.nn.Module):
    def __init__(self, num_classes, C=32):
        super(Net, self).__init__()

        self.stem_mlp = MLP([3, C])

        self.sa1_module = SAModule(0.02, 0.04, 32, [C + 3, C * 2, C * 4])
        self.sa2_module = SAModule(0.04, 0.08, 32, [C * 4 + 3, C * 6, C * 8])
        self.sa3_module = SAModule(0.08, 0.16, 32, [C * 8 + 3, C * 12, C * 16])

        self.fp3_module = FPModule(2, [C * 24, C * 20, C * 16])
        self.fp2_module = FPModule(2, [C * 20, C * 18, C * 16])
        self.fp1_module = FPModule(2, [C * 17, C * 16, C * 16])

        # self.fp3_module = FPModule(3, [C * 24, C * 12, C * 6])
        # self.fp2_module = FPModule(3, [C * 10, C * 5, C * 3])
        # self.fp1_module = FPModule(3, [C * 4, C * 2, C])

        self.conv1 = torch.nn.Conv1d(C * 16, C * 16, 1)
        self.conv2 = torch.nn.Conv1d(C * 16, num_classes, 1)
        self.norm = GN(select_num_groups(C*16), C * 16)

        # self.conv1 = torch.nn.Conv1d(C, C, 1)
        # self.conv2 = torch.nn.Conv1d(C, num_classes, 1)
        # self.norm = GN(select_num_groups(C), C)

        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, data):

        data.x = self.stem_mlp(data.pos[:,:3])
        sa0_out = (data.x, data.pos, data.reflectance, data.batch, data.sf)

        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)

        fp3_out = self.fp3_module(*sa3_out[:4], *sa2_out[:4])
        fp2_out = self.fp2_module(*fp3_out, *sa1_out[:4])
        x, _, refl, _ = self.fp1_module(*fp2_out, *sa0_out[:4])
        
        x = self.conv1(x.unsqueeze(dim=0).permute(0, 2, 1))
        x = F.relu(self.norm(x), inplace=True)
        x = torch.squeeze(self.conv2(x)).to(torch.float)

        return x

