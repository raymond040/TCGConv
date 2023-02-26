#Importing Packages
import os
from platform import node
import sys
import os.path as osp
import copy

import torch
from torch.nn import Linear, Parameter
from torch_geometric.utils import degree
from torch.nn import LSTM
from torch_geometric.nn.aggr import Aggregation
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
import torchmetrics
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
from torch_scatter import scatter

sys.path.insert(0, '/hpctmp/e0407728/FYP/Ver1-26Jul/utils/')
from TimeHorizonExperiment import Time_Batches
from config import parse_args
from util import setup_device
from DatasetPrep import DatasetPrep
from models import focal_loss
from LineGraph_Transform import LineGraph_Transform

from typing import Dict, List, Optional, Tuple
from torch_geometric.typing import Adj, OptTensor
import pandas as pd
   

class TemporalHeteroHG(torch.nn.Module):
    """
    This class initialises the TemporalGCNConv Layer 
    Inputs:
    in_channels = The dimension of the attribute of the nodes
    hidden_channels = The number of hidden channels set arbitrarily
    out_channels = The number of output, should be = 1
    num_layers = The number of layers, set arbitrarily
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, fc_hid):
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
        self.dropout = nn.Dropout(p=0.5)

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
        return output[:,-1]

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
            output = out + self.bias #
            outout.append(output)
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
        #Sort based on to_node because we assume each node has been sorted by time. so this input is eligible for LSTM
        index, indices_sorted_based_on_to_node_column = index.sort(dim=-1,descending=False, stable=True)
        inputs = inputs[indices_sorted_based_on_to_node_column]

        out_lstm = self.lstm_aggr(x=inputs, index=index,
                ptr=ptr, dim_size=dim_size,
                dim= -2)
                
        return out_lstm

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

    F1 = F1_Score(preds=preds_ArgMax, target=labels)[1].to(args.device)

    accuracy = (confmat[0][0] + confmat[1][1]) / torch.sum(confmat)
    recall = confmat[1][1] / (confmat[1][1] + confmat[1][0])
    precision = confmat[1][1] / (confmat[1][1] + confmat[0][1])

    output['AP'] = AP.item()
    output['R'] = recall.item()
    output['P'] = precision.item()
    output['F1'] = F1.item()
    output['loss'] = loss.item()
    output['conf'] = confmat

    if epoch % args.num_epochs_print ==0:
        print(mode+"_loss:", loss)
        print(mode+"_accuracy", accuracy)
        print(mode+"_recall", recall)
        print(mode+"_precision", precision)
        print(mode+"_AP", AP)
        print(mode+"_F1",F1)
        print(confmat)

    return output

def saveModel(args, model, optimizer, F1, AP, conf, Precision, Recall, path):
    state={'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'F1': F1,
    'AP': AP,
    'Precision':Precision,
    'Recall':Recall,
    'conf':conf,
    'seed': args.seed,
    }
    torch.save(state,path)

def CG_trainer(args,config,Train_Batches, Test_Batches):
    best_all_dct = {
        'F1':[],
        'AP':[],
        'P':[],
        'R':[],
        'model_loc': []
    }
    i = 0
    for batch1, batch2 in zip(Train_Batches, Test_Batches):
        print(i)
        if args.type_ED == "sub":
            if i == 0:
                train_batch = copy.deepcopy(batch1).to(args.device)
                test_batch = copy.deepcopy(next(Train_Batches)).to(args.device)

                # initiate model
                nodetype = train_batch.node_types[0]
                initial_data = train_batch.to(args.device)
                inpt_chnl = train_batch[nodetype].x.shape[1]
                model = TemporalHeteroHG(inpt_chnl,config["hidden_chnl"],config["hidden_chnl"],config["num_layers"], config["hidden_chnl"]).to(args.device)

                with torch.no_grad():  # Initialize lazy modules.
                    out = model(initial_data[nodetype].x, initial_data[nodetype].ts, initial_data.edge_index_dict).to(args.device)

                optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
                loss_fn = focal_loss(alpha=config['alpha'], gamma=config['gamma'], num_classes=2, size_average=True)

            elif i == args.num_batches - 1:
                break

            else:
                train_batch = copy.deepcopy(test_batch).to(args.device)
                test_batch = copy.deepcopy(batch1).to(args.device)
        elif args.type_ED == "TT":
            if i == 0:
                train_batch = copy.deepcopy(batch1).to(args.device)
                test_batch = copy.deepcopy(batch2).to(args.device)

                # initiate model
                nodetype = train_batch.node_types[0]
                initial_data = train_batch.to(args.device)
                inpt_chnl = train_batch[nodetype].x.shape[1]
                model = TemporalHeteroHG(inpt_chnl,config["hidden_chnl"],config["hidden_chnl"],config["num_layers"], config["hidden_chnl"]).to(args.device)

                with torch.no_grad():  # Initialize lazy modules.
                    out = model(initial_data[nodetype].x, initial_data[nodetype].ts, initial_data.edge_index_dict).to(args.device)

                optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
                loss_fn = focal_loss(alpha=config['alpha'], gamma=config['gamma'], num_classes=2, size_average=True)
            elif i == args.num_batches - 1:
                break
            else:
                train_batch = copy.deepcopy(batch1).to(args.device)
                test_batch = copy.deepcopy(batch2).to(args.device)
        
        print ('================================================================')
        print ('This is batch: ' + str(i))
        best_each_batch_dct = {
            'F1':0,
            'AP':0,
            'P':0,
            'R':0,
            'conf':None,
            'model':None,
            'model_loc':0,
            'opt_state_dict':None
        }
        for epoch in range(1,args.num_epochs+1):
            loss = train(model=model, graph=train_batch, optimizer = optimizer, loss_fn = loss_fn )
            train_loss = test(args=args, model=model, graph=train_batch, loss_fn=loss_fn, mode = 'Train', epoch = epoch)['loss']
            test_output = test(args=args, model=model, graph=test_batch, loss_fn=loss_fn, mode = 'Test', epoch = epoch)

            test_loss = test_output['loss']
            current_epoch_F1, current_epoch_AP = test_output['F1'],test_output['AP']
            if current_epoch_F1 > best_each_batch_dct['F1']:
                best_each_batch_dct['F1'] = copy.deepcopy(test_output['F1'])
                best_each_batch_dct['AP'] = copy.deepcopy(test_output['AP'])
                best_each_batch_dct['P'] = copy.deepcopy(test_output['P'])
                best_each_batch_dct['R'] = copy.deepcopy(test_output['R'])
                best_each_batch_dct['conf'] = copy.deepcopy(test_output['conf'])
                best_each_batch_dct['model'] = copy.deepcopy(model)
                best_each_batch_dct['model_loc'] = epoch
                best_each_batch_dct['opt_state_dict'] = copy.deepcopy(optimizer.state_dict())
            elif current_epoch_F1 == best_each_batch_dct['F1']:
                if current_epoch_AP > best_each_batch_dct['AP']:
                    best_each_batch_dct['F1'] = copy.deepcopy(test_output['F1'])
                    best_each_batch_dct['AP'] = copy.deepcopy(test_output['AP'])
                    best_each_batch_dct['P'] = copy.deepcopy(test_output['P'])
                    best_each_batch_dct['R'] = copy.deepcopy(test_output['R'])
                    best_each_batch_dct['conf'] = copy.deepcopy(test_output['conf'])
                    best_each_batch_dct['model'] = copy.deepcopy(model)
                    best_each_batch_dct['model_loc'] = epoch
                    best_each_batch_dct['opt_state_dict'] = copy.deepcopy(optimizer.state_dict())

            if epoch % args.num_epochs_print == 0:
                print(f'Epoch: {epoch:03d}, Loss: {loss:.6f}, Train: {train_loss:.6f}, Test: {test_loss:.6f}')  

        #################### finish iteration for all epochs #################

        #Passing the model and optimizer to the next batch
        model = best_each_batch_dct['model'].to(args.device)
        opt_state_dict = best_each_batch_dct['opt_state_dict']
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
        optimizer.load_state_dict(opt_state_dict)

        # Save to pth for pytorch readability
        modelPath = os.path.join(args.root, 'Results/'+args.dataset_name+'/Baseline3B_'+args.dataset_name+'_'+args.type_ED+'_'+str(args.num_version)+'.pth')
        saveModel(
            args,
            model,
            optimizer,
            best_each_batch_dct['F1'],
            best_each_batch_dct['AP'],
            best_each_batch_dct['conf'],
            best_each_batch_dct['P'],
            best_each_batch_dct['R'],
            modelPath)

        ############################ Storing in big dictionary too for human readability #########################
        best_all_dct['F1'].append(best_each_batch_dct['F1'])
        best_all_dct['AP'].append(best_each_batch_dct['AP'])
        best_all_dct['P'].append(best_each_batch_dct['P'])
        best_all_dct['R'].append(best_each_batch_dct['R'])
        best_all_dct['model_loc'].append(best_each_batch_dct['model_loc'])

        df = pd.DataFrame.from_dict(best_all_dct) 
        df.to_csv(args.csvPath, index=False, header=True)

        i = i + 1
        ################################ End of all batches #################################################################

def main():
    args = parse_args()
    setup_device(args)
    args.csvPath = os.path.join(args.root, 'Results/'+args.dataset_name+'/Baseline3B_'+args.dataset_name+'_'+args.type_ED+'_'+str(args.num_version)+'.csv')

    config = {
            "alpha": 0.0423596838172482,
            "gamma": 2,
            "lr": 1e-4,
            "hidden_chnl": 32,
            "dropout": 0.2,
            "num_layers": 3,
        }
    
    if args.dataset_name =="CC":
        args.root_readData = args.root+"Credit_Card_Fraud"
    elif args.dataset_name == "MOOC":
        args.root_readData = args.root+"MOOC"
    else:
        raise Exception("Unknown dataset_name")

    # data preparation
    Graph = DatasetPrep(dataset_name=args.dataset_name, root=args.root_readData, type=args.graph_type, percentage = args.percentage)[0]
    G_Batches,G_Test = Time_Batches(Graph, edge_temporal = True, num_batches = args.num_batches, HeteroEdges = False, Test_Batch = True)
    CG_Batches = map(LineGraph_Transform, G_Batches)
    CG_Test = map(LineGraph_Transform, G_Test)

    if args.type_ED == "TT":
        CG_trainer(args, config, CG_Batches, CG_Test)
    elif args.type_ED == "sub":
        CG_trainer(args, config, CG_Batches, copy.deepcopy(CG_Batches))
    else:
        raise Exception("type_ED is invalid, only sub and TT are supported.")


if __name__ == '__main__':
    main()
    
