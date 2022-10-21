import os
import sys
import copy
import time
import pandas as pd

import torch
from torch import Tensor
import torchmetrics
import torch.nn as nn
from torch.nn import LSTM, Linear, Parameter
import torch.nn.functional as F

from torch_geometric.utils import degree
from torch_geometric.nn.aggr import Aggregation
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter

from typing import Optional, Tuple


sys.path.insert(0, '/workspaces/Edge-Representation-Learning-in-Temporal-Graph/utils')
from util import saveModel,focal_loss,load_checkpoint
   

class TemporalHeteroHG(torch.nn.Module):
    """
    This class initialises the TemporalGCNConv Layer 
    Inputs:
    in_channels = The dimension of the attribute of the nodes
    hidden_channels = The number of hidden channels set arbitrarily
    out_channels = The number of output, should be = 1
    num_layers = The number of layers, set arbitrarily
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, fc_hid,dropout):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers): #Must do sth here so that the input channel is 1 every convolution
            if _ == 0:
                conv = TemporalGCNConv(in_channels=in_channels, out_channels=hidden_channels)
            elif _ == num_layers-1:
                conv = TemporalGCNConv(in_channels=hidden_channels, out_channels=out_channels)
            else:
                conv = TemporalGCNConv(in_channels=hidden_channels, out_channels=hidden_channels)
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels, bias=False)
        self.fc = nn.Sequential(
                        nn.Linear(out_channels, fc_hid),
                        nn.ReLU(),
                        nn.Linear(fc_hid,2)
                        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, node_ts, edge_index_dict):
        for conv in self.convs:
            x = conv(x, node_ts, edge_index_dict)
        x = self.dropout(x)
        x = self.fc(x)
        return x 

class LSTMAggregation(Aggregation):
    r"""Performs LSTM-style aggregation in which the elements to aggregate are
    interpreted as a sequence.
    .. warn::
        :class:`LSTMAggregation` is not permutation-invariant.
    .. note::
        :class:`LSTMAggregation` requires sorted indices. Here the chosen sorting mechanism: Sort by to_node (recipient) indices.
        The nodes, are however, required to be pre-processed such that sorting of nodes by index corresponds sorting it by its time.
    Args:
        in_channels (int): Size of each input sample. 
        out_channels (int): Size of each output sample. 
        **kwargs (optional): Additional arguments of :class:`torch.nn.LSTM`.
    """
    def __init__(self, in_channels: int, out_channels: int, **kwargs): 
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lstm = LSTM(in_channels, out_channels, batch_first=True, **kwargs) 
        self.reset_parameters() 

    def reset_parameters(self):
        self.lstm.reset_parameters() 

    def to_dense_batch(self, x: Tensor, index: Optional[Tensor] = None,
                       ptr: Optional[Tensor] = None,
                       dim_size: Optional[int] = None,
                       dim: int = -2) -> Tuple[Tensor, Tensor]:
 
        self.assert_index_present(index) 
        self.assert_sorted_index(index) #ensure sorted
        self.assert_two_dimensional_input(x, dim) 
        return to_dense_batch(x, index, batch_size=dim_size,fill_value=-999.0) 

    def forward(self, x: Tensor, index: Optional[Tensor] = None, 
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:
        x, _ = self.to_dense_batch(x, index, ptr, dim_size, dim) 
        output = self.lstm(x)[0]
        outputs = output[:,-1]
        return outputs

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels})')

class TemporalGCNConv(MessagePassing):
    """
    This class peforms the convolution, message passing. 
    Inputs:
    x = node features of dim N x Number of Inpt Channels
    node_ts = time stamp of the nodes
    edge_index_dict = a dictionary of two ways of edge indexes for heterogenity.
    The forward pass will be run for each direction of edge indexes.

    Here, before message is passed to LSTM Aggregation, message is ensured to be eligible
    This is done by finding the eligible edge index.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')  
        self.lin_dict = torch.nn.ModuleDict()
        self.lin = Linear(in_channels, out_channels, bias=False)
        self.bias = Parameter(torch.Tensor(out_channels))
        self.aggregators = ['sum']#, 'mean', 'min', 'max', 'var', 'std']
        self.scalers = ['identity']#, 'amplification', 'attenuation', 'linear', 'inverse_linear']
        self.lstm_aggr = LSTMAggregation(in_channels,out_channels)
        self.reset_parameters()

    def reset_parameters(self): 
        self.lin.reset_parameters()
        self.bias.data.zero_()
    
    def forward(self, x, node_ts, edge_index_dict):

        # x has shape [N, in_channels], edge_index has shape [2, E]

        node_ts = node_ts.reshape(-1,1) # in case the dimension is not of shape [N,1]

        outout=[] #container for output from both directions (Heterogenous)

        count = 0
        for _,edge_idx_type in edge_index_dict.items(): #iterate for each edge type, do the forward pass.

            edge_index = edge_idx_type

            # Vanilla Normalisation Technique 
            fromnode, tonode = edge_index
            deg = degree(tonode, x.size(0), dtype=x.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            norm = deg_inv_sqrt[fromnode] * deg_inv_sqrt[tonode]

            #Finding eligible edge indexes
            eligible_edge_mask = (node_ts[fromnode]<=node_ts[tonode]).reshape((-1,)) 
            
            out = self.propagate(edge_index[:,eligible_edge_mask], x=x, norm=norm, node_ts = node_ts)
            outs = out + self.bias #
            outout.append(outs)
            count = count + 1

        return sum(outout)/count

    def message(self,x_j, x_i, norm, node_ts_j, node_ts_i): 
        r"""Constructs messages from node :math:`j` to node :math:`i`
        in analogy to :math:`\phi_{\mathbf{\Theta}}` for each edge in
        :obj:`edge_index`.
        This function can take any argument as input which was initially
        passed to :meth:`propagate`.
        Furthermore, tensors passed to :meth:`propagate` can be mapped to the
        respective nodes :math:`i` and :math:`j` by appending :obj:`_i` or
        :obj:`_j` to the variable name, *.e.g.* :obj:`x_i` and :obj:`x_j`.
        """
        return x_j

    def aggregate(self, inputs: Tensor, index: Tensor, node_ts: Tensor,
                  ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None) -> Tensor:
        r"""Aggregates messages from neighbors as
        :math:`\square_{j \in \mathcal{N}(i)}`.

        Takes in the output of message computation as first argument and any
        argument which was initially passed to :meth:`propagate`.

        By default, this function will delegate its call to scatter functions
        that support "add", "mean", "min", "max" and "mul" operations as
        specified in :meth:`__init__` by the :obj:`aggr` argument.
        """
        inputs = self.lin(inputs)
        outs = [] 

        for aggregator in self.aggregators:
            if aggregator == 'sum':
                out = scatter(inputs, index, 0, None, dim_size, reduce='sum')
            elif aggregator == 'mean':
                out = scatter(inputs, index, 0, None, dim_size, reduce='mean')
            elif aggregator == 'min':
                out = scatter(inputs, index, 0, None, dim_size, reduce='min')
            elif aggregator == 'max':
                out = scatter(inputs, index, 0, None, dim_size, reduce='max')
            elif aggregator == 'var' or aggregator == 'std':
                mean = scatter(inputs, index, 0, None, dim_size, reduce='mean')
                mean_squares = scatter(inputs * inputs, index, 0, None,
                                       dim_size, reduce='mean')
                out = mean_squares - mean * mean
                if aggregator == 'std':
                    out = torch.sqrt(torch.relu(out) + 1e-5)
            else:
                raise ValueError(f'Unknown aggregator "{aggregator}".')
            outs.append(out) 

        out = torch.cat(outs, dim=-1) 
        deg = degree(index, dim_size, dtype=inputs.dtype)
        deg = deg.clamp_(1).view(-1, 1, 1) 

        outs = []

        for scaler in self.scalers:
            if scaler == 'identity':
                pass
            elif scaler == 'amplification':
                out = out * (torch.log(deg + 1) / self.avg_deg['log'])
            elif scaler == 'attenuation':
                out = out * (self.avg_deg['log'] / torch.log(deg + 1))
            elif scaler == 'linear':
                out = out * (deg / self.avg_deg['lin'])
            elif scaler == 'inverse_linear':
                out = out * (self.avg_deg['lin'] / deg)
            else:
                raise ValueError(f'Unknown scaler "{scaler}".')
            outs.append(out) 

        output = torch.cat(outs,dim=-1)
        return output

    def update(self, inputs: Tensor) -> Tensor:
        r"""Updates node embeddings in analogy to
        :math:`\gamma_{\mathbf{\Theta}}` for each node
        :math:`i \in \mathcal{V}`.
        Takes in the output of aggregation as first argument and any argument
        which was initially passed to :meth:`propagate`.
        """
        return inputs

def train(model, graph, optimizer, loss_fn):
    nodetype = graph.node_types[0]
    model.train()
    optimizer.zero_grad()
    logits = model(graph[nodetype].x, graph[nodetype].ts, graph.edge_index_dict)
    labels = graph[nodetype].y
    loss = loss_fn(logits, labels)
    loss.backward()
    optimizer.step()
    return float(loss)

@torch.no_grad()
def test(args, model, graph, loss_fn, mode, epoch):
    nodetype = graph.node_types[0]
    output = {}

    model.eval()
    logits = model(graph[nodetype].x, graph[nodetype].ts, graph.edge_index_dict)
    labels = graph[nodetype].y

    if mode == 'Test' and args.type_ED == "TT":
        mask = graph[nodetype].testgraph_mask
        labels = labels[mask]
        logits = logits[mask]

    preds = F.softmax(logits, dim=1)
    preds_ArgMax = preds.argmax(dim=1)

    loss = loss_fn(logits, labels)

    confmat = torchmetrics.functional.confusion_matrix(preds=preds, target=labels.int(), num_classes=2,
                                                        normalize=None, threshold=0.5, multilabel=False).to(args.device)

    AP = torchmetrics.functional.average_precision(preds=preds, target=labels, num_classes=2, pos_label=1).to(args.device)

    F1_Score = torchmetrics.F1Score(num_classes=2, average=None).to(args.device)

    if torch.isnan(AP).item(): # use f1 as proxy if AP is NAN
        print("WARNING: AP is NAN, use F1 instead")
        AP = F1_Score(preds=preds, target=labels).to(args.device)

    F1 = F1_Score(preds=preds_ArgMax, target=labels)[1]

    accuracy = (confmat[0][0] + confmat[1][1]) / torch.sum(confmat)
    recall = confmat[1][1] / (confmat[1][1] + confmat[1][0])
    precision = confmat[1][1] / (confmat[1][1] + confmat[0][1])

    output['AP'] = AP.item()
    output['R'] = recall.item()
    output['P'] = precision.item()
    output['F1'] = F1.item()
    output['loss'] = loss.item()
    output['conf'] = confmat

    if epoch % args.num_epochs_print == 0:
        print(mode+"_loss:", round(loss.item(),4))
        print(mode+"_accuracy", round(accuracy.item(),4))
        print(mode+"_recall", round(recall.item(),4))
        print(mode+"_precision", round(precision.item(),4))
        print(mode+"_AP", round(AP.item(),4))
        print(mode+"_F1",round(F1.item(),4))
        print(confmat)

    return output

def CGConv_sum_Trainer(args,config,Train_Groups, Test_Groups):
    i = 0
    t0 = time.time()

    if args.mode == 'test':
        for group1 in Train_Groups:
            test_group = group1
        nodetype = test_group.node_types[0]
        inpt_chnl = test_group[nodetype].x.shape[1]

        model = TemporalHeteroHG(inpt_chnl,128, 128, 2, 128,config["dropout"]).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay']) 
        loss_fn = focal_loss(alpha=config['alpha'], gamma=config['gamma'], num_classes=2, size_average=True)

        filename = os.path.join(args.root,'Results/'+args.dataset_name+'/'+args.model_type+'_pretrained.pth')
        model,optimizer = load_checkpoint(args,filename, model, optimizer)
        _ = test(args = args, model = model, graph = test_group.to(args.device), loss_fn = loss_fn, epoch = args.num_epochs_print, mode = 'Test')
        print("finished loading pretrained model and test for the last group")

    elif args.mode =='train':
        for group1, group2 in zip(Train_Groups, Test_Groups):
            t_group = time.time()
            if args.type_ED == "sub":
                if i == 0:
                    train_group = copy.deepcopy(group1).to(args.device)
                    test_group = copy.deepcopy(next(Train_Groups)).to(args.device)

                    # initiate model
                    nodetype = train_group.node_types[0]
                    initial_data = train_group.to(args.device)
                    inpt_chnl = train_group[nodetype].x.shape[1]
                    model = TemporalHeteroHG(inpt_chnl,config["hidden_chnl"],config["hidden_chnl"],config["num_layers"], config["hidden_chnl"],config["dropout"]).to(args.device)

                    with torch.no_grad():  # Initialize lazy modules.
                        out = model(initial_data[nodetype].x, initial_data[nodetype].ts, initial_data.edge_index_dict).to(args.device)

                    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
                    loss_fn = focal_loss(alpha=config['alpha'], gamma=config['gamma'], num_classes=2, size_average=True)
                    best_all_dct = {
                        'AP':[],
                        'P':[],
                        'R':[],
                        'F1':[],
                        'model' : [model],
                        'model_loc': [],
                        'optimizer': [optimizer],
                        'optimizer_dict': [optimizer.state_dict()],
                    }

                elif i == args.num_groups - 1:
                    break

                else:
                    train_group = copy.deepcopy(test_group).to(args.device)
                    test_group = copy.deepcopy(group1).to(args.device)
            elif args.type_ED == "TT":
                if i == 0:
                    train_group = copy.deepcopy(group1).to(args.device)
                    test_group = copy.deepcopy(group2).to(args.device)

                    # initiate model
                    nodetype = train_group.node_types[0]
                    initial_data = train_group.to(args.device)
                    inpt_chnl = train_group[nodetype].x.shape[1]
                    model = TemporalHeteroHG(inpt_chnl,config["hidden_chnl"],config["hidden_chnl"],config["num_layers"], config["hidden_chnl"],config["dropout"]).to(args.device)

                    with torch.no_grad():  # Initialize lazy modules.
                        out = model(initial_data[nodetype].x, initial_data[nodetype].ts, initial_data.edge_index_dict).to(args.device)

                    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
                    loss_fn = focal_loss(alpha=config['alpha'], gamma=config['gamma'], num_classes=2, size_average=True)

                    best_all_dct = {
                        'AP':[],
                        'P':[],
                        'R':[],
                        'F1':[],
                        'model' : [model],
                        'model_loc': [],
                        'optimizer': [optimizer],
                        'optimizer_dict': [optimizer.state_dict()],
                    }
                elif i == args.num_groups - 1:
                    break
                else:
                    train_group = copy.deepcopy(group1).to(args.device)
                    test_group = copy.deepcopy(group2).to(args.device)
            
            print ('================================================================')
            print ('This is group: ' + str(i))
            best_each_group_dct = {
                'F1':0,
                'AP':0,
                'P':0,
                'R':0,
                'conf':None,
                'model':None,
                'model_loc':0,
                'optimizer': None,
                'optimizer_dict':None
            }

            optimizer = torch.optim.Adam(best_all_dct['model'][i].parameters(), lr=config["lr"], weight_decay=config['weight_decay'])
            optimizer.load_state_dict(best_all_dct['optimizer_dict'][i])

            for epoch in range(1,args.num_epochs+1):
                t_epoch = time.time()

                # First model is the initial model, and first optimizer is initial optimizer
                # At the end of each group, we will store the best epoch's model and optimizer that generates largest F1
                loss = train(model=best_all_dct['model'][i], graph=train_group, optimizer = optimizer, loss_fn = loss_fn )
                train_loss = test(args=args, model=best_all_dct['model'][i], graph=train_group, loss_fn=loss_fn, mode = 'Train', epoch = epoch)['loss']
                test_output = test(args=args, model=best_all_dct['model'][i], graph=test_group, loss_fn=loss_fn, mode = 'Test', epoch = epoch)

                test_loss = test_output['loss']
                current_epoch_F1, current_epoch_AP = test_output['F1'],test_output['AP']
                if current_epoch_F1 > best_each_group_dct['F1']:
                    best_each_group_dct['F1'] = copy.deepcopy(test_output['F1'])
                    best_each_group_dct['AP'] = copy.deepcopy(test_output['AP'])
                    best_each_group_dct['P'] = copy.deepcopy(test_output['P'])
                    best_each_group_dct['R'] = copy.deepcopy(test_output['R'])
                    best_each_group_dct['conf'] = copy.deepcopy(test_output['conf'])
                    best_each_group_dct['model'] = copy.deepcopy(model)
                    best_each_group_dct['model_loc'] = epoch
                    best_each_group_dct['optimizer_dict'] = copy.deepcopy(optimizer.state_dict())
                elif current_epoch_F1 == best_each_group_dct['F1']:
                    if current_epoch_AP > best_each_group_dct['AP']:
                        best_each_group_dct['F1'] = copy.deepcopy(test_output['F1'])
                        best_each_group_dct['AP'] = copy.deepcopy(test_output['AP'])
                        best_each_group_dct['P'] = copy.deepcopy(test_output['P'])
                        best_each_group_dct['R'] = copy.deepcopy(test_output['R'])
                        best_each_group_dct['conf'] = copy.deepcopy(test_output['conf'])
                        best_each_group_dct['model'] = copy.deepcopy(model)
                        best_each_group_dct['model_loc'] = epoch
                        best_each_group_dct['optimizer_dict'] = copy.deepcopy(optimizer.state_dict())

                if epoch % args.num_epochs_print == 0:
                    print(f'Epoch: {epoch:03d}, Loss: {loss:.6f}, Train: {train_loss:.6f}, Test: {test_loss:.6f}, Time per epoch: {(time.time() - t_epoch):.4f}')

                if epoch == args.num_epochs:
                    if best_each_group_dct['model'] == None:
                        print('WARNING: Current group has ALL 0 F1 and AP (Model = None Type), pass current model to next group')
                        best_each_group_dct['model'] = copy.deepcopy(best_all_dct['model'][i])
                        best_each_group_dct['optimizer_dict'] = copy.deepcopy(best_all_dct['optimizer_dict'][i])   

            ############################ Storing best results per group in big dictionary #########################
            best_all_dct['AP'].append(best_each_group_dct['AP'])
            best_all_dct['P'].append(best_each_group_dct['P'])
            best_all_dct['R'].append(best_each_group_dct['R'])
            best_all_dct['F1'].append(best_each_group_dct['F1'])
            best_all_dct['model'].append(copy.deepcopy(best_each_group_dct['model']))
            best_all_dct['optimizer_dict'].append(copy.deepcopy(best_each_group_dct['optimizer_dict']))
            best_all_dct['model_loc'].append(best_each_group_dct['model_loc'])

            # write best results per group in csv instead of write them all when all groups are finished!
            df = pd.DataFrame.from_dict({k:best_all_dct[k] for k in ('AP','P','R','F1','model_loc') if k in best_all_dct})
            df.to_csv(args.csvPath, index=False, header=True)

            # print time per group
            print (f'Time per group: {(time.time()-t_group):.4f}')

            # for last group, save model
            if i == args.num_groups - 2:
                optimizer = torch.optim.Adam(best_all_dct['model'][-1].parameters(), lr=config["lr"], weight_decay=config['weight_decay'])
                optimizer.load_state_dict(best_all_dct['optimizer_dict'][-1])
                saveModel(args, best_all_dct['model'][-1],optimizer,best_all_dct['F1'][-1],best_all_dct['AP'][-1],best_all_dct['P'][-1],best_all_dct['R'][-1], args.modelPath)

            i += 1

        # print average results and time for the whole training
        print("Finished training!")
        avg_AP = df['AP'].mean()
        avg_F1 = df['F1'].mean()
        avg_P = df['P'].mean()
        avg_R = df['R'].mean()
        print(f'Average AP: {avg_AP:.4f}, Average F1: {avg_F1:.4f}, Average Precision: {avg_P:.4f}, Average Recall: {avg_R:.4f}, Total Time: {(time.time() - t0):.4f} ')
