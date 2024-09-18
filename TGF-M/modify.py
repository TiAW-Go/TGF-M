import numpy as np
import torch as pt
import torch.nn as nn
import torch.nn.parameter as nnp
import torch.nn.functional as nnf
from sklearn.manifold import TSNE
from torch_scatter import scatter
from torch_geometric.utils import degree
from torch_geometric.nn import MessagePassing
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
import matplotlib.pyplot as plt

MAX_DEGREE = 4

def plot_heatmap(data, log_transform=False, vmin=None, vmax=None):
    data = data.detach().cpu().numpy()
    if log_transform:
        data = np.log(data + 1e-6)
    plt.figure(figsize=(10, 10))
    heatmap = plt.imshow(data, cmap='hot', vmin=vmin, vmax=vmax, interpolation='nearest')

    plt.colorbar(heatmap)
    plt.title("Heatmap with Optional Log Transform and Custom Colormap")
    plt.show()

# DeepNet: https://arxiv.org/abs/2203.00555v1
class ScaleLayer(nn.Module):
    def __init__(self, width, scale_init):
        super().__init__()
        self.scale = nnp.Parameter(pt.zeros(width) + np.log(scale_init))

    def forward(self, x):
        return pt.exp(self.scale) * x

# Graphormer: https://arxiv.org/abs/2106.05234v5
class ScaleDegreeLayer(nn.Module):
    def __init__(self, width, scale_init):
        super().__init__()
        self.scale = nnp.Parameter(pt.zeros(MAX_DEGREE, width) + np.log(scale_init))

    def forward(self, x, deg):
        return pt.exp(self.scale)[deg] * x

# GLU: https://arxiv.org/abs/1612.08083v3
class GatedLinearBlock(nn.Module):
    def __init__(self, width, num_head, scale_act, dropout=0.1, block_name=None):
        super().__init__()

        self.pre   = nn.Sequential(nn.Conv1d(width, width, 1),
                         nn.GroupNorm(num_head, width, affine=False))
        self.gate  = nn.Sequential(nn.Conv1d(width, width*scale_act, 1, bias=False, groups=num_head),
                         nn.ReLU(), nn.Dropout(dropout))
        self.value = nn.Conv1d(width, width*scale_act, 1, bias=False, groups=num_head)
        self.post  = nn.Conv1d(width*scale_act, width, 1)
        if block_name is not None:
            print('##params[%s]:' % block_name, np.sum([np.prod(p.shape) for p in self.parameters()]))

    def forward(self, x):
        xx = self.pre(x.unsqueeze(-1))
        xx = self.gate(xx) * self.value(xx)
        xx = self.post(xx).squeeze(-1)
        return xx

class GatedLinearBlock2(nn.Module):
    def __init__(self, width, num_head, scale_act, dropout=0.1):
        super().__init__()

        self.gate  = nn.Sequential(nn.GroupNorm(num_head, width, affine=False),
                         nn.Conv1d(width, width*scale_act, 1, bias=False, groups=num_head),
                         nn.ReLU(), nn.Dropout(dropout))
        self.value = nn.Sequential(nn.GroupNorm(num_head, width, affine=False),
                         nn.Conv1d(width, width*scale_act, 1, bias=False, groups=num_head))
        self.post  = nn.Conv1d(width*scale_act, width, 1)

    def forward(self, xg, xv):
        xx = self.gate(xg.unsqueeze(-1)) * self.value(xv.unsqueeze(-1))
        xx = self.post(xx).squeeze(-1)
        return xx

# VoVNet: https://arxiv.org/abs/1904.09730v1
class ConvMessage(MessagePassing):
    def __init__(self, width, width_head, width_scale, hop, kernel, scale_init=0.1):
        super().__init__(aggr="add")
        self.width = width
        self.hop = hop

        self.bond_encoder = nn.ModuleList()
        self.pre = nn.ModuleList()
        self.msg = nn.ModuleList()
        self.scale_2D = nn.ModuleList()

        for _ in range(hop*kernel):
            self.bond_encoder.append(BondEncoder(emb_dim=width))
            self.pre.append(nn.Linear(width, width, bias=False))
            self.msg.append(GatedLinearBlock2(width, width_head, width_scale))
            self.scale_2D.append(ScaleDegreeLayer(width, scale_init))


        print('##params[conv]:', np.sum([np.prod(p.shape) for p in self.parameters()]))

    def forward(self, x, node_degree, edge_index, edge_attr, mode):
        for layer in range(len(self.msg)):
            if layer == 0:
                x_raw, x_out = x, 0
            elif layer % self.hop == 0:
                x_raw, x_out = x + x_out, 0

            x_raw = self.propagate(edge_index, x=self.pre[layer](x_raw), edge_attr=edge_attr, layer=layer)
            x_out = x_out + self.scale_2D[layer](x_raw, node_degree)

        return x_out

    def message(self, x_i, x_j, edge_attr, layer):
        bond = self.bond_encoder[layer](edge_attr)
        msg = self.msg[layer](x_i + bond, x_j + bond)
        return msg
        
    def update(self, aggr_out):
        return aggr_out

# GIN-virtual: https://arxiv.org/abs/2103.09430
class VirtMessage(nn.Module):
    def __init__(self, width, width_head, width_scale, scale_init=0.01):
        super().__init__()
        self.width = width

        self.msg = GatedLinearBlock(width, width_head, width_scale)
        self.scale = ScaleLayer(width, scale_init)
        print('##params[virt]:', np.sum([np.prod(p.shape) for p in self.parameters()]))

    def forward(self, x, x_res, batch, batch_size):
        xx = x_res = scatter(x, batch, dim=0, dim_size=batch_size, reduce='sum') + x_res
        xx = self.scale(self.msg(xx))[batch]
        return xx, x_res

# CosFormer: https://openreview.net/pdf?id=Bl8CQrx2Up4
class AttMessage(nn.Module):
    def __init__(self, width, width_head, width_scale, scale_init=0.01):
        super().__init__()
        self.width = width
        self.width_head = width_head

        num_grp = width // width_head
        self.pre  = nn.Sequential(nn.Conv1d(width, width, 1),
                         nn.GroupNorm(num_grp, width, affine=False))
        self.msgq = nn.Conv1d(width, width*width_scale, 1, bias=False, groups=num_grp)
        self.msgk = nn.Conv1d(width, width*width_scale, 1, bias=False, groups=num_grp)
        self.msgv = nn.Conv1d(width, width*width_scale, 1, bias=False, groups=num_grp)
        self.post = nn.Conv1d(width*width_scale, width, 1)
        self.scale = ScaleLayer(width, scale_init)
        print('##params[att]:', np.sum([np.prod(p.shape) for p in self.parameters()]))

    def forward(self, x, x_res, batch, batch_size):
        xv = self.pre(x.unsqueeze(-1))

        shape = [len(x), -1, self.width_head]
        xq = pt.exp(self.msgq(xv) / np.sqrt(self.width_head)).reshape(shape)
        xk = pt.exp(self.msgk(xv) / np.sqrt(self.width_head)).reshape(shape)
        xv = self.msgv(xv).reshape(shape)

        xv = pt.einsum('bnh,bnv->bnhv', xk, xv)
        xv = x_res = scatter(xv, batch, dim=0, dim_size=batch_size, reduce='sum') + x_res
        xk = scatter(xk, batch, dim=0, dim_size=batch_size, reduce='sum')[batch]
        xq = xq / pt.einsum('bnh,bnh->bn', xq, xk)[:, :, None]  # norm
        xv = pt.einsum('bnh,bnhv->bnv', xq, xv[batch]).reshape(len(x), -1, 1)

        xv = self.scale(self.post(xv).squeeze(-1))
        return xv, x_res

@pt.jit.script
def gaussian(x, mean, std):
    pi = 3.14159
    a = (2*pi) ** 0.5
    return pt.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)

class GaussianLayer(nn.Module):
    def __init__(self, K=128, edge_types=11):
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)
        nn.init.uniform_(self.means.weight, 2, 3)
        nn.init.uniform_(self.stds.weight, 1, 1.5)


    def forward(self, x, edge_types):
        # mul = self.mul(edge_types)
        # bias = self.bias(edge_types)
        # x = mul * x.unsqueeze(-1) + bias
        x = x.unsqueeze(-1)
        mean = self.means.weight.float().view(-1).unsqueeze(0)
        std = (self.stds.weight.float().view(-1).abs() + 1e-2).unsqueeze(0)
        x = gaussian(x.float(), mean, std).type_as(self.means.weight)
        return x

class DistanceSumScalingLayer(nn.Module):
    def __init__(self, feature_dim, scale_init=0.001):
        super(DistanceSumScalingLayer, self).__init__()
        bins= [-1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 170, 190, 224]
        self.bins = pt.tensor(bins, dtype=pt.float32).to('cuda:0')
        self.scale = nn.Parameter(pt.ones(len(bins)-1, feature_dim) * scale_init)

    def forward(self, x, distance_sum):
        bin_indices = pt.bucketize(distance_sum, self.bins) - 1
        scaling_factors = pt.exp(-self.scale[bin_indices] * distance_sum.unsqueeze(1))
        return x * scaling_factors

class AttentionModule(nn.Module):
    def __init__(self, emb_dim):
        super(AttentionModule, self).__init__()
        self.emb_dim = emb_dim
        self.attention_weights = nn.Linear(2 * emb_dim, 1)

    def forward(self, h_2d, h_3d):
        combined = pt.cat([h_2d, h_3d], dim=-1)
        attention_scores = self.attention_weights(combined)
        attention_weights = pt.sigmoid(attention_scores)
        print("attention_weights:",attention_weights)
        fused_output = attention_weights * h_2d + (1 - attention_weights) * h_3d
        return fused_output.squeeze(-1)

class TGF_M(nn.Module):
    def __init__(self, num_layers, emb_dim, conv_hop, conv_kernel, use_virt=True, use_att=True, JK=None, gnn_type=None):
        super().__init__()
        self.num_layers = num_layers
        self.atom_encoder = AtomEncoder(emb_dim)
        self.bond_encoder_3D = GaussianLayer(emb_dim)
        self.bond_encoder_2D = BondEncoder(emb_dim)
        self.bond_encoder_both = AttentionModule(emb_dim)
        self.scale_both = ScaleDegreeLayer(emb_dim, 0.1)

        self.conv = pt.nn.ModuleList()
        self.virt = pt.nn.ModuleList()
        self.att = pt.nn.ModuleList()
        self.main = pt.nn.ModuleList()
        for layer in range(num_layers):
            self.conv.append(ConvMessage(emb_dim, 16, 1, conv_hop, conv_kernel))
            self.virt.append(VirtMessage(emb_dim, 16, 2) if use_virt else None)
            self.att.append(AttMessage(emb_dim, 16, 2) if use_att else None)
            self.main.append(GatedLinearBlock(emb_dim, 16, 3))  # debug

    def forward(self, batched_data, mode):
        x, edge_index, edge_attr, batch, edge_distance, distance_sum = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch, batched_data.edge_distance, batched_data.sum_distance
        edge_types = batched_data.edge_types
        batch_size = len(batched_data.ptr) - 1
        node_degree = degree(edge_index[1], len(x)).long() - 1
        node_degree.clamp_(0, MAX_DEGREE - 1)
        if mode == '2D':
            h_in = self.bond_encoder_2D(edge_attr)

        elif mode == '3D':
            h_in = self.bond_encoder_3D(edge_distance, edge_types)

        elif mode == 'both':
            h_in = self.bond_encoder_both(self.bond_encoder_2D(edge_attr),self.bond_encoder_3D(edge_distance, edge_types))

        h_in = scatter(h_in, edge_index[1], dim=0, dim_size=len(x), reduce='sum')
        h_in = self.scale_both(h_in, node_degree)
        h_in, h_att, h_virt = self.atom_encoder(x) + h_in, 0, 0

        for layer in range(self.num_layers):
            h_out = h_in + self.conv[layer](h_in, node_degree, edge_index, edge_attr, mode)
            if self.virt[layer] is not None:
                h_tmp, h_virt = self.virt[layer](h_in, h_virt, batch, batch_size)
                h_out, h_tmp = h_out + h_tmp, None
            if self.att[layer] is not None:
                h_tmp, h_att = self.att[layer](h_in, h_att, batch, batch_size)
                h_out, h_tmp = h_out + h_tmp, None
            h_out = h_in = self.main[layer](h_out)

        return h_out