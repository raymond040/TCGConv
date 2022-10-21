import os
import sys
import copy
import time
import pandas as pd
import torch
import torch.nn
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv, to_hetero

sys.path.insert(0, '/workspaces/Edge-Representation-Learning-in-Temporal-Graph/utils')
from util import saveModel,focal_loss,load_checkpoint

class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            if _ == num_layers-1:
                conv = SAGEConv((-1, -1), out_channels)
            else:
                conv = SAGEConv((-1, -1), hidden_channels)
            self.convs.append(conv)
        self.num_layers = num_layers

    def forward(self, x, edge_index):
        count = 0
        for conv in self.convs:
            if count == self.num_layers - 1:
                x = conv(x, edge_index)
            else:
                x = conv(x, edge_index).relu()
            count += 1
        return x
class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels, nodetypes, dropout):
        super().__init__()
        self.fc=nn.Sequential(nn.LazyLinear(hidden_channels),
                              nn.ReLU(),
                              nn.LazyLinear(hidden_channels),
                              nn.ReLU(),
                              nn.LazyLinear(2))
        self.dropout = nn.Dropout(p=dropout)
        self.nodetypes = nodetypes

    def forward(self, z_dict, edge_label_index,edge_attr):
        row, col = edge_label_index # row is edge_label_index[0], 0-th row
        z = torch.cat([z_dict[self.nodetypes[0]][row], z_dict[self.nodetypes[1]][col],edge_attr], dim=-1) # get user and activity's corresponding features
        z = self.dropout(z)
        z = self.fc(z)
        return z

class ConcatModel(torch.nn.Module):
    def __init__(self, hidden_channels, nodetypes,init_metadata,dropout, num_layers):
        super().__init__()
        self.encoder = GNNEncoder(hidden_channels, hidden_channels, num_layers)
        self.encoder = to_hetero(self.encoder, init_metadata, aggr='sum') 
        self.decoder = EdgeDecoder(hidden_channels,nodetypes,dropout)

    def forward(self, x_dict, edge_index_dict, edge_label_index, edge_attr):
        z_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(z_dict, edge_label_index, edge_attr)

def train(graph, model, optimizer, loss_fn):
    edgetypes = graph.metadata()[-1]
    model.train()
    optimizer.zero_grad()
    logits = model(graph.x_dict, graph.edge_index_dict, graph[edgetypes[0]].edge_label_index, graph[edgetypes[0]].edge_attr)
    target = graph[edgetypes[0]].edge_label
    loss = loss_fn(logits, target)
    loss.backward()
    optimizer.step()
    return float(loss)

@torch.no_grad()
def test(args, model, graph, loss_fn, epoch, mode = 'Train'):
    output={}
    edgetypes = graph.metadata()[-1]
    model.eval()
    logits = model(graph.x_dict, graph.edge_index_dict, graph[edgetypes[0]].edge_label_index, graph[edgetypes[0]].edge_attr)
    labels = graph[edgetypes[0]].edge_label

    if mode == 'Test' and args.type_ED == "TT":
        mask = graph[edgetypes[0]].edge_testgraph_mask
        labels = labels[mask]
        logits = logits[mask]

    preds = F.softmax(logits,dim=1)
    preds2 = preds.argmax(dim=1)
    loss = loss_fn(preds, labels)
    confmat = torchmetrics.functional.confusion_matrix(preds=preds, target=labels, num_classes=2,
                                                        normalize=None, threshold=0.5, multilabel=False).to(args.device)
    AP = torchmetrics.functional.average_precision(preds=preds, target=labels, num_classes=2, pos_label=None).to(args.device)
    F1_Score = torchmetrics.F1Score(num_classes=2,average=None)[1].to(args.device)
    F1 = F1_Score(preds2,labels)

    if torch.isnan(AP).item():
        #use f1 as proxy if AP is NAN
        print("WARNING: AP is NAN, use F1 instead")
        F1_Score = torchmetrics.F1Score(num_classes=2,average=None)[1].to(args.device)
        AP = F1_Score(preds2,labels)
        # raise Exception("AP is NAN, no positive samples")

    accuracy = (confmat[0][0] + confmat[1][1]) / torch.sum(confmat)
    recall = confmat[1][1] / (confmat[1][1] + confmat[1][0])
    precision = confmat[1][1] / (confmat[1][1] + confmat[0][1])

    output['AP'] = AP.item()
    output['R'] = recall.item()
    output['P'] = precision.item()
    output['F1'] = F1.item()
    output['loss'] = loss.item()

    if epoch % args.num_epochs_print == 0:
        print(mode+"_loss:", round(loss.item(),4))
        print(mode+"_accuracy", round(accuracy.item(),4))
        print(mode+"_recall", round(recall.item(),4))
        print(mode+"_precision", round(precision.item(),4))
        print(mode+"_AP", round(AP.item(),4))
        print(mode+"_F1",round(F1.item(),4))
        print(confmat)

    return output

def CONCAT_Trainer(args,config,Train_Groups, Test_Groups):
    nodetypes = Train_Groups[0].metadata()[0]

    # init_graph is for initiate model
    init_graph = T.ToUndirected()(copy.deepcopy(Train_Groups[0]))
    relationship = init_graph.metadata()[-1][0]
    init_graph[relationship].edge_label_index = init_graph[relationship].edge_index
    loss_fn = focal_loss(alpha=config["alpha"], gamma=config["gamma"], num_classes=2, size_average=True)

    if args.mode == 'test':
        model = ConcatModel(256, nodetypes, init_graph.metadata(),config["dropout"], config["num_layers"]).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay']) 
        filename = os.path.join(args.root,'Results/'+args.dataset_name+'/'+args.model_type+'_pretrained.pth')
        model,optimizer = load_checkpoint(args,filename, model, optimizer)
        test_batch = T.ToUndirected()(Train_Groups[args.num_groups-1])
        test_batch[relationship].edge_label_index = test_batch[relationship].edge_index
        _ = test(args = args, model = model, graph = test_batch.to(args.device), loss_fn = loss_fn, epoch = args.num_epochs_print, mode = 'Test')
        print("finished loading pretrained model and test for the last group")

    elif args.mode =='train':
        model = ConcatModel(config["hidden_chnl"], nodetypes, init_graph.metadata(),config["dropout"], config["num_layers"]).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config['weight_decay'])
    
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
        t0 = time.time()
        for batch in range(args.num_groups-1):
            t_batch = time.time()
            
            # different test data per experimental design 
            if Test_Groups is None:
                test_batch = T.ToUndirected()(Train_Groups[batch+1])
                test_batch[relationship].edge_label_index = test_batch[relationship].edge_index
            else:
                test_batch = Test_Groups[batch]
                test_batch = T.ToUndirected()(test_batch)
                test_batch[relationship].edge_label_index = test_batch[relationship].edge_index
            train_batch = T.ToUndirected()(Train_Groups[batch])
            train_batch[relationship].edge_label_index = train_batch[relationship].edge_index
            
            print ('=========================================================')
            print ('This is batch: ' + str(batch))
            best_each_batch_dct = {
                'AP':0,
                'P':0,
                'R':0,
                'F1':0,
                'model':None,
                'model_loc': 0,
                'optimizer': None,
                'optimizer_dict':None
            }
            optimizer = torch.optim.Adam(best_all_dct['model'][batch].parameters(), lr=config["lr"], weight_decay=config['weight_decay'])
            optimizer.load_state_dict(best_all_dct['optimizer_dict'][batch])

            for epoch in range(1,args.num_epochs+1):
                t_epoch = time.time()

                # First model is the initial model, and first optimizer is initial optimizer
                # At the end of each batch, we will store the best epoch's model and optimizer that generates largest F1
                loss = train(model = best_all_dct['model'][batch], graph = train_batch.to(args.device), optimizer = optimizer, loss_fn = loss_fn )
                train_loss = test(args = args, model = best_all_dct['model'][batch], graph = train_batch.to(args.device), loss_fn=loss_fn, epoch = epoch, mode = 'Train')['loss']
                test_output = test(args = args, model = best_all_dct['model'][batch], graph = test_batch.to(args.device), loss_fn = loss_fn, epoch = epoch, mode = 'Test')
                test_loss = test_output['loss']
                current_epoch_f1 = test_output['F1']

                if current_epoch_f1 > best_each_batch_dct['F1']:
                    best_each_batch_dct['AP'] = test_output['AP']
                    best_each_batch_dct['P'] = test_output['P']
                    best_each_batch_dct['R'] = test_output['R']
                    best_each_batch_dct['F1'] = current_epoch_f1
                    best_each_batch_dct['model'] = copy.deepcopy(model)
                    best_each_batch_dct['model_loc'] = epoch
                    best_each_batch_dct['optimizer_dict'] = copy.deepcopy(optimizer.state_dict())
                if epoch % args.num_epochs_print == 0:
                    print(f'Epoch: {epoch:03d}, Loss: {loss:.6f}, Train: {train_loss:.6f}, Test: {test_loss:.6f}, Time per epoch: {(time.time() - t_epoch):.4f}')
                if epoch == args.num_epochs:
                    if best_each_batch_dct['model'] == None:
                        print('WARNING: Current batch has ALL 0 F1 and AP (Model = None Type), pass current model to next batch')
                        best_each_batch_dct['model'] = copy.deepcopy(best_all_dct['model'][batch])
                        best_each_batch_dct['optimizer_dict'] = copy.deepcopy(best_all_dct['optimizer_dict'][batch])  


            best_all_dct['AP'].append(best_each_batch_dct['AP'])
            best_all_dct['P'].append(best_each_batch_dct['P'])
            best_all_dct['R'].append(best_each_batch_dct['R'])
            best_all_dct['F1'].append(best_each_batch_dct['F1'])
            best_all_dct['model'].append(copy.deepcopy(best_each_batch_dct['model']))
            best_all_dct['optimizer_dict'].append(copy.deepcopy(best_each_batch_dct['optimizer_dict']))
            best_all_dct['model_loc'].append(best_each_batch_dct['model_loc'])

            if batch > 2:
                best_all_dct['model'][batch-2] = None
                best_all_dct['optimizer_dict'][batch-2] = None

            # write best results per batch in csv instead of write them all when all groups are finished!
            df = pd.DataFrame.from_dict({k:best_all_dct[k] for k in ('AP','P','R','F1','model_loc') if k in best_all_dct})
            df.to_csv(args.csvPath, index=False, header=True)

            # print time per batch
            print (f'Time per batch: {(time.time()-t_batch):.4f}')

            # for last batch, save model
            if batch == args.num_groups - 2:
                optimizer = torch.optim.Adam(best_all_dct['model'][-1].parameters(), lr=config["lr"], weight_decay=config['weight_decay'])
                optimizer.load_state_dict(best_all_dct['optimizer_dict'][-1])
                saveModel(args, best_all_dct['model'][-1],optimizer,best_all_dct['F1'][-1],best_all_dct['AP'][-1],best_all_dct['P'][-1],best_all_dct['R'][-1], args.modelPath)

        # print average results and time for the whole training
        print("Finished training!")
        avg_AP = df['AP'].mean()
        avg_F1 = df['F1'].mean()
        avg_P = df['P'].mean()
        avg_R = df['R'].mean()
        print(f'Average AP: {avg_AP:.4f}, Average F1: {avg_F1:.4f}, Average Precision: {avg_P:.4f}, Average Recall: {avg_R:.4f}, Total Time: {(time.time() - t0):.4f} ')