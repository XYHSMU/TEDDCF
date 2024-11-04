import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pandas as pd
import sys
from ISTF_attention import SelfAttentionLayer


class GLU(nn.Module):
    def __init__(self, features, dropout=0.1):#PEMS08: 192
        super(GLU, self).__init__()
        self.conv1 = nn.Conv2d(features, features, (1, 1))
        self.conv2 = nn.Conv2d(features, features, (1, 1))
        self.conv3 = nn.Conv2d(features, features, (1, 1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):


        x1 = self.conv1(x)
        x2 = self.conv2(x)
        out = x1 * torch.sigmoid(x2)
        out = self.dropout(out)
        out = self.conv3(out)
        return out#[64,192,170,12]


class TemporalEmbedding(nn.Module):
    def __init__(self, time, features):
        super(TemporalEmbedding, self).__init__()
        #S08:time 288 features 96
        self.time = time
        self.time_day = nn.Parameter(torch.empty(time, features))#[288 96]
        nn.init.xavier_uniform_(self.time_day)

        self.time_week = nn.Parameter(torch.empty(7, features))#[7 96]
        nn.init.xavier_uniform_(self.time_week)

    def forward(self, x):
        #x #in:[64,12,170,3]
        day_emb = x[..., 1]

        time_day = self.time_day[(day_emb[:, :, :] * self.time).type(torch.LongTensor)]


        time_day = time_day.transpose(1, 2).contiguous()

        week_emb = x[..., 2]


        time_week = self.time_week[(week_emb[:, :, :]).type(torch.LongTensor)]#[64,12,170,96]
        time_week = time_week.transpose(1, 2).contiguous()#torch.Size([64, 170, 12, 96])


        tem_emb = time_day + time_week#[64,170,12,96]

        tem_emb = tem_emb.permute(0,3,1,2)#[64,96,170,12]

        return tem_emb


    
class Diffusion_GCN(nn.Module):
    def __init__(self, channels=128, diffusion_step=1, dropout=0.1):
        super().__init__()
        self.diffusion_step = diffusion_step#1
        self.conv = nn.Conv2d(diffusion_step * channels, channels, (1, 1))#[192,192,(1,1)]
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):

        out = []
        for i in range(0, self.diffusion_step):#1
            if adj.dim() == 3:
                x = torch.einsum("bcnt,bnm->bcmt", x, adj).contiguous()
                out.append(x)
            elif adj.dim() == 2:
                x = torch.einsum("bcnt,nm->bcmt", x, adj).contiguous()
                out.append(x)
        x = torch.cat(out, dim=1)
        x = self.conv(x)
        output = self.dropout(x)
        return output

    
class EventGraph_Fusion(nn.Module):
    def __init__(self, channels=128, num_nodes=170, diffusion_step=1, dropout=0.1):
        super().__init__()
        self.memory = nn.Parameter(torch.randn(channels, num_nodes))
        nn.init.xavier_uniform_(self.memory)
        self.fc = nn.Linear(2,1)
        
    def forward(self, x):
        adj_dyn_1 = torch.softmax(
            F.relu(
                torch.einsum("bcnt, cm->bnm", x, self.memory).contiguous()
                / math.sqrt(x.shape[1])
            ),
            -1,
        )
        adj_dyn_2 = torch.softmax(
            F.relu(
                torch.einsum("bcn, bcm->bnm", x.sum(-1), x.sum(-1)).contiguous()
                / math.sqrt(x.shape[1])
            ),
            -1,
        )
        adj_f = torch.cat([(adj_dyn_1).unsqueeze(-1)] + [(adj_dyn_2).unsqueeze(-1)], dim=-1)

        adj_f = torch.softmax(self.fc(adj_f).squeeze(), -1)

        topk_values, topk_indices = torch.topk(adj_f, k=int(adj_f.shape[1]*0.8), dim=-1)

        mask = torch.zeros_like(adj_f)

        mask.scatter_(-1, topk_indices, 1)

        adj_f = adj_f * mask

        return adj_f


class EventGCN(nn.Module):
    def __init__(self, channels=128, num_nodes=170, diffusion_step=1, dropout=0.1, emb=None):
        super().__init__()

        self.conv = nn.Conv2d(channels,channels,(1,1))
        self.generator = EventGraph_Fusion(channels, num_nodes, diffusion_step, dropout)
        self.gcn = Diffusion_GCN(channels, diffusion_step, dropout)
        self.emb = emb

    def forward(self, x):

        skip = x
        x = self.conv(x)
        adj_dyn = self.generator(x)
        x = self.gcn(x, adj_dyn)
        x = x*self.emb + skip

        return x

class TrendGCN(nn.Module):
    def __init__(self, channels=128, num_nodes=170, diffusion_step=1, dropout=0.1, emb=None):
        super().__init__()

        self.conv = nn.Conv2d(channels,channels,(1,1))
        self.generator = TrendGraph_Fusion(channels, num_nodes, diffusion_step, dropout)
        self.gcn = Diffusion_GCN(channels, diffusion_step, dropout)
        self.emb = emb


    def forward(self, x):

        skip = x
        x = self.conv(x)
        adj_dyn = self.generator(x)
        x = self.gcn(x, adj_dyn)
        x = x*self.emb + skip

        return x


class TrendGraph_Fusion(nn.Module):
    def __init__(self, channels=128, num_nodes=170, diffusion_step=1, dropout=0.1):
        super().__init__()
        self.memory = nn.Parameter(
            torch.randn(channels, num_nodes))
        nn.init.xavier_uniform_(self.memory)
        self.fc = nn.Linear(2, 1)
        self.E_adaptive = nn.Parameter(torch.randn(num_nodes, 10))

    def forward(self, x):
        # adj_dyn_1 = torch.softmax(
        #     F.relu(
        #         torch.einsum("bcnt, cm->bnm", x, self.memory).contiguous()
        #         / math.sqrt(x.shape[1])
        #     ),
        #     -1,
        # )

        adj_dyn_2 = torch.softmax(
            F.relu(
                torch.einsum("bcn, bcm->bnm", x.sum(-1), x.sum(-1)).contiguous()
                / math.sqrt(x.shape[1])
            ),
            -1,
        )
        adj_adp = F.softmax(F.relu(torch.mm(self.E_adaptive, self.E_adaptive.transpose(0, 1))), dim=1)

        adj_adp_expanded = adj_adp.unsqueeze(0)

        adj_adp = adj_adp_expanded.repeat(x.shape[0], 1, 1)

        adj_f = torch.cat([(adj_dyn_2).unsqueeze(-1)] + [(adj_adp).unsqueeze(-1)], dim=-1)

        adj_f = torch.softmax(self.fc(adj_f).squeeze(), -1)

        topk_values, topk_indices = torch.topk(adj_f, k=int(adj_f.shape[1] * 0.8), dim=-1)

        mask = torch.zeros_like(adj_f)

        mask.scatter_(-1, topk_indices, 1)

        adj_f = adj_f * mask

        return adj_f

class Chomp1d(nn.Module):
    """
    extra dimension will be added by padding, remove it
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :, :-self.chomp_size].contiguous()

class TemporalConvNet(nn.Module):
    def __init__(self, features, kernel_size=2, dropout=0.2, levels=1):
        super(TemporalConvNet, self).__init__()

        layers = []
        for i in range(levels):
            dilation_size = 2 ** i
            padding = (kernel_size - 1) * dilation_size
            self.conv = nn.Conv2d(features, features, (1, kernel_size), dilation=(1, dilation_size),
                                  padding=(0, padding))
            self.chomp = Chomp1d(padding)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)

            layers += [nn.Sequential(self.conv, self.chomp, self.relu, self.dropout)]
        self.tcn = nn.Sequential(*layers)

    def forward(self, xh):
        xh = self.tcn(xh)
        return xh
    pass

class FeedForward(nn.Module):
    def __init__(self, fea, res_ln=False):
        super(FeedForward, self).__init__()



        self.res_ln = res_ln
        self.L = len(fea) - 1#2
        self.linear = nn.ModuleList([nn.Linear(fea[i], fea[i+1]) for i in range(self.L)])
        self.ln = nn.LayerNorm(fea[self.L], elementwise_affine=False)

    def forward(self, inputs):

        x = inputs
        for i in range(self.L):
            x = self.linear[i](x)
            if i != self.L-1:
                x = F.relu(x)


        if self.res_ln:
            x += inputs
            x = self.ln(x)
        return x

class Adaptive_Fusion(nn.Module):
    def __init__(self, heads, dims):
        super(Adaptive_Fusion, self).__init__()
        features = dims  # 192
        self.h = heads  # 8
        self.d = int(dims / heads) # 16

        self.qlfc = FeedForward([features, features])
        self.khfc = FeedForward([features, features])
        self.vhfc = FeedForward([features, features])
        self.ofc = FeedForward([features, features])

        self.ln = nn.LayerNorm(features, elementwise_affine=False)
        self.ff = FeedForward([features, features, features], True)

    def forward(self, xl, xh, Mask=True):
        '''
        xl: [B,T,N,F]
        xh: [B,T,N,F]
        te: [B,T,N,F]
        return: [B,T,N,F]
        '''
        # xl += te
        # xh += te

        query = self.qlfc(xl)  # [B,T,N,F]
        keyh = torch.relu(self.khfc(xh))  # [B,T,N,F]
        valueh = torch.relu(self.vhfc(xh))  # [B,T,N,F]

        query = torch.cat(torch.split(query, self.d, -1), 0).permute(0, 2, 1, 3)  # [k*B,N,T,d]
        keyh = torch.cat(torch.split(keyh, self.d, -1), 0).permute(0, 2, 3, 1)  # [k*B,N,d,T]
        valueh = torch.cat(torch.split(valueh, self.d, -1), 0).permute(0, 2, 1, 3)  # [k*B,N,T,d]

        attentionh = torch.matmul(query, keyh)  # [k*B,N,T,T]

        if Mask:
            batch_size = xl.shape[0]
            num_steps = xl.shape[1]
            num_vertexs = xl.shape[2]
            mask = torch.ones(num_steps, num_steps).to(xl.device)  # [T,T]
            mask = torch.tril(mask)  # [T,T]
            mask = torch.unsqueeze(torch.unsqueeze(mask, dim=0), dim=0)  # [1,1,T,T]
            mask = mask.repeat(self.h * batch_size, num_vertexs, 1, 1)  # [k*B,N,T,T]
            mask = mask.to(torch.bool)
            zero_vec = (-2 ** 15 + 1) * torch.ones_like(attentionh).to(xl.device)  # [k*B,N,T,T]
            attentionh = torch.where(mask, attentionh, zero_vec)

        attentionh /= (self.d ** 0.5)  # scaled
        attentionh = F.softmax(attentionh, -1)  # [k*B,N,T,T]

        value = torch.matmul(attentionh, valueh)  # [k*B,N,T,d]

        value = torch.cat(torch.split(value, value.shape[0] // self.h, 0), -1).permute(0, 2, 1, 3)  # [B,T,N,F]
        value = self.ofc(value)
        value = value + xl

        value = self.ln(value)

        return self.ff(value)  # [64,12,170,128]

class TEDDCF(nn.Module):
    def __init__(
        self, device, input_dim, num_nodes, channels, granularity, dropout=0.1
    ):
        super().__init__()

        self.device = device
        self.num_nodes = num_nodes
        self.output_len = 12
        self.input_len = 12
        self.heads = 8
        diffusion_step = 1

        self.Temb = TemporalEmbedding(granularity, channels)

        self.start_conv = nn.Conv2d(
            in_channels=input_dim, out_channels=channels, kernel_size=(1, 1)
        )


        self.glu = GLU(channels*2, dropout)

        self.regression_layer = nn.Conv2d(
            channels*2, self.output_len, kernel_size=(1, self.output_len)
        )

        self.temporal_conv = TemporalConvNet(channels*2)
        self.pre_h = nn.Conv2d(in_channels=self.input_len, out_channels=self.output_len, kernel_size=(1,1))
        self.adp_f = Adaptive_Fusion(self.heads, channels*2)

        num_layers = 3
        self.attn_layers_t = nn.ModuleList(
            [
                SelfAttentionLayer(channels*2, feed_forward_dim=256, num_heads=4, dropout=0.1)
                for _ in range(num_layers)  # 3
            ]
        )
        self.xh_emb = nn.Parameter(torch.randn(channels*2, num_nodes, 12))
        self.xh_dgcn = EventGCN(channels*2, num_nodes, diffusion_step=1, dropout=0.1,emb=self.xh_emb)

        self.xl_emb = nn.Parameter(torch.randn(channels*2, num_nodes, 12))
        self.xl_dgcn = TrendGCN(channels*2, num_nodes, diffusion_step=1, dropout=0.1, emb=self.xl_emb)


    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])

    def forward(self, inputxl, inputxh):

        xl = inputxl
        xh = inputxh

        # Encoder
        # Data Embedding
        time_embl = self.Temb(inputxl.permute(0, 3, 2, 1))
        time_embh = self.Temb(inputxh.permute(0, 3, 2, 1))
        #t = self.start_conv(x)#[64,96,170,12]
        xl = torch.cat([self.start_conv(xl)] + [time_embl], dim=1)
        xh = torch.cat([self.start_conv(xh)] + [time_embh], dim=1)



        xl = xl.permute(0, 3, 2, 1)
        for attn in self.attn_layers_t:
            xl = attn(xl, dim=1)
        xl = xl.permute(0, 3, 2, 1)

        xl = self.xl_dgcn(xl)
        xl = self.glu(xl) + xl


        xh = self.temporal_conv(xh)


        xh = self.xh_dgcn(xh)

        #simple plus
        x_all = xh + xl
        #STwave_fusion
        # xl = xl.transpose(1, 3)
        # xh = self.pre_h(xh.transpose(1,3))#[64,12,170,192]
        # x_all = self.adp_f(xl, xh)
        # x_all = x_all.transpose(1, 3)

        prediction = self.regression_layer(F.relu(x_all))


        return prediction
