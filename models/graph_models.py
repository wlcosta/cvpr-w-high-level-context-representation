import dgl
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GINConv
from dgl.nn.pytorch.glob import SumPooling, AvgPooling

class MLP(nn.Module):
    """Construct two-layer MLP-type aggreator for GIN model"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linears = nn.ModuleList()
        # two-layer MLP
        self.linears.append(nn.Linear(input_dim, hidden_dim, bias=False))
        self.linears.append(nn.Linear(hidden_dim, output_dim, bias=False))
        self.batch_norm = nn.BatchNorm1d((hidden_dim))

    def forward(self, x):
        h = x
        h = F.relu(self.batch_norm(self.linears[0](h)))
        return self.linears[1](h)

class GIN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, pooling_type='SumPooling'):
        super().__init__()
        self.ginlayers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.pooling_type = pooling_type
        num_layers = 5

        for layer in range(num_layers - 1):
            if layer == 0:
                mlp = MLP(in_feats, h_feats, h_feats)
            else:
                mlp = MLP(h_feats, h_feats, h_feats)
            
            self.ginlayers.append(
                GINConv(mlp, learn_eps=False)
            )
            self.batch_norms.append(nn.BatchNorm1d(h_feats))

        self.linear_prediction = nn.ModuleList()
        self.vad_prediction = nn.ModuleList()

        for layer in range(num_layers):
            if layer == 0:
                self.linear_prediction.append(nn.Linear(in_feats, num_classes))
                self.vad_prediction.append(nn.Linear(in_feats, 3))
            else:
                self.linear_prediction.append(nn.Linear(h_feats, num_classes))
                self.vad_prediction.append(nn.Linear(h_feats, 3))

        self.drop = nn.Dropout(0.5)
        if self.pooling_type == 'SumPooling':
            self.pool = (
                SumPooling()
            )
        elif self.pooling_type == 'AvgPooling':
            self.pool = (
                AvgPooling()
            )

    def forward(self, g, h):
        hidden_rep = [h]
        for i, layer in enumerate(self.ginlayers):
            h = layer(g, h)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            hidden_rep.append(h)

        score_over_layer = 0
        score_over_vad = 0
        for i, h in enumerate(hidden_rep):
            pooled_h = self.pool(g, h)
            score_over_layer += self.drop(self.linear_prediction[i](pooled_h))
            score_over_vad += self.drop(self.vad_prediction[i](pooled_h))

        return score_over_layer, score_over_vad

class DefaultGCNModel(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(DefaultGCNModel, self).__init__()
        self.conv1 = dgl.nn.GraphConv(in_feats, h_feats)
        self.conv2 = dgl.nn.GraphConv(h_feats, h_feats)
        self.classifier_cat = nn.Linear(h_feats, num_classes)
        self.classifier_cont = nn.Linear(h_feats, 3)
    
    def forward(self, g, in_feat):
        h = F.relu(self.conv1(g, in_feat))
        h = F.relu(self.conv2(g, h))

        with g.local_scope():
            g.ndata['h'] = h
            hg = dgl.mean_nodes(g, 'h')
            return self.classifier_cat(hg), self.classifier_cont(hg)

    
def get_gcn_model(model_name):
    if model_name == 'DefaultGCNModel':
        return DefaultGCNModel()
    