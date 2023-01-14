import os
import sys
import copy
import time
import pandas as pd
import torchmetrics
import torch.nn
import torch.nn as nn
from torch_geometric.nn import HGTConv, Linear
from torchmetrics import AveragePrecision, F1Score, Precision, Recall, AUROC

HPC_Flag = True
if HPC_Flag:
    sys.path.insert(0, '/home/svu/e0407728/My_FYP/TCGConv/utils')
else:
    sys.path.insert(0, '/workspaces/TCGConv/utils')
from util import saveModel,focal_loss,load_checkpoint
  
class HGT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers, initialGraph, dropout):
        super().__init__()
            
        # For each node type, we have a separate linear layer
        self.lin_dict = torch.nn.ModuleDict()
        for node_type in initialGraph.node_types:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        # We use one HGTConv layer
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, initialGraph.metadata(), num_heads, group='sum')
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x_dict, edge_index_dict):
        x_dict = {
            node_type: self.lin_dict[node_type](x).relu_()
            for node_type, x in x_dict.items()
        }
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        x = self.dropout(x_dict['trans_node'])
        return torch.nn.functional.normalize(torch.nn.LeakyReLU(0.2)(self.lin(x)), 
        p = 2, dim = 1, eps = 1e-12, out = None)

def train(graph, model, optimizer, loss_fn):
    model.train()
    optimizer.zero_grad()
    out = model(graph.x_dict, graph.edge_index_dict)
    loss = loss_fn(out, graph['trans_node'].y)
    loss.backward()
    optimizer.step()
    return float(loss)

@torch.no_grad()
def test(args, model, graph, loss_fn, epoch,  mode = 'Train'):
    nodetype = graph.node_types[-1]
    output={}

    model.eval()
    pred = model(graph.x_dict, graph.edge_index_dict)
    labels = graph[nodetype].y

    if mode == 'Test' and args.type_ED == "TT":
        mask = graph[nodetype].testgraph_mask
        labels = labels[mask]
        pred = pred[mask]

    # SoftMax for AP and ArgMax for precision, recall and F1 
    softMaxPred = pred.softmax(dim = 1) #preds (pred = logits)
    argMaxPred = softMaxPred.argmax(dim = 1) #predsArgMax
    loss = loss_fn(softMaxPred, labels)
    
    # Define evaluation functions
    confmat = torchmetrics.functional.confusion_matrix(preds=softMaxPred, target=labels, num_classes=2,
                                                        normalize=None, threshold=0.5, multilabel=False).to(args.device)
    average_precision = AveragePrecision(num_classes = 2, pos_label=1).to(args.device)
    accuracy = (confmat[0][0] + confmat[1][1]) / torch.sum(confmat)
    precision = Precision(num_classes = 2, average = 'none').to(args.device)
    recall = Recall(num_classes = 2, average = 'none').to(args.device)
    f1 = F1Score(num_classes = 2, average = 'none').to(args.device)
    auroc = AUROC(num_classes = 2).to(args.device)

    # For computing evaluation scores
    apScore = average_precision(softMaxPred, labels)
    precisionScore = precision(argMaxPred, labels)
    precisionScore =  precisionScore[1:]
    recallScore = recall(argMaxPred, labels)
    recallScore =  recallScore[1:]
    f1Score = f1(argMaxPred, labels)
    f1Score = f1Score[1:]
    aurocScore = auroc(softMaxPred, labels)

    if torch.isnan(apScore).item():
        #use f1 as proxy if AP is NAN
        print("WARNING: AP is NAN, use F1 instead")
        f1Score = torchmetrics.F1Score(num_classes=2,average=None)[1].to(args.device)
        apScore = f1(argMaxPred,labels)
        # raise Exception("AP is NAN, no positive samples")

    output['AP'] = apScore.item()
    output['R'] = recallScore.item()
    output['P'] = precisionScore.item()
    output['F1'] = f1Score.item()
    output['AUROC'] = aurocScore.item()
    output['loss'] = loss.item()

    if epoch % args.num_epochs_print == 0:
        print(mode+"_loss:", round(loss.item(),4))
        print(mode+"_accuracy", round(accuracy.item(),4))
        print(mode+"_recall", round(recallScore.item(),4))
        print(mode+"_precision", round(precisionScore.item(),4))
        print(mode+"_AP", round(apScore.item(),4))
        print(mode+"_F1",round(f1Score.item(),4))
        print(mode+"_AUROC",round(aurocScore.item(),4))
        print(confmat)

    return output

def ATN_Trainer(args,config,Train_Groups, Test_Groups):
    init_graph = Train_Groups[0].to(args.device)
    model = HGT(hidden_channels=256, out_channels=2, num_heads=1, num_layers=config['num_layers'], initialGraph= init_graph, dropout=config['dropout']).to(args.device)
    loss_fn = focal_loss(alpha = config['alpha'], gamma = config['gamma'])   

    if args.mode == 'test':
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay']) 
        filename = os.path.join(args.root,'Results/'+args.dataset_name+'/'+args.model_type+'_pretrained.pth')
        model,optimizer = load_checkpoint(args,filename,model, optimizer)
        test_group = Train_Groups[args.num_groups-1].to(args.device)
        _ = test(args = args, model = model, graph = test_group, loss_fn = loss_fn, epoch = args.num_epochs_print, mode = 'Test')
        print("finished loading pretrained model and test for the last group")

    elif args.mode =='train':
        with torch.no_grad():  # Initialize lazy modules.
            out = model(init_graph.x_dict, init_graph.edge_index_dict).to(args.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])    

        best_all_dct = {
            'AP':[],
            'P':[],
            'R':[],
            'F1':[],
            'AUROC':[],
            #'model' : [model],
            'model_loc': [],
            #'optimizer': [optimizer],
            #'optimizer_dict': [optimizer.state_dict()],
        }

        t0 = time.time()
        for i in range(args.num_groups - 1):
            none_models = False
            none_models_lst = []
            t_batch = time.time()

            # different test data per experimental design 
            if Test_Groups is None: # sub
                test_group = Train_Groups[i+1].to(args.device)
            else:
                test_group = Test_Groups[i].to(args.device)
            train_group = Train_Groups[i].to(args.device)    

            print ('=========================================================')
            print ('This is group: ' + str(i)) 

            best_each_group_dct = {
                'AP':-1,
                'P':-1,
                'R':-1,
                'F1':-1,
                'AUROC':0,
                'model':None,
                'model_loc': 0,
                'optimizer': None,
                'optimizer_dict':None
            }

            optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config['weight_decay'])
            optimizer.load_state_dict(optimizer.state_dict())

            for epoch in range(1, args.num_epochs + 1):
                t_epoch = time.time()

                # First model is the initial model, and first optimizer is initial optimizer
                # At the end of each batch, we will store the best epoch's model and optimizer that generates largest F1
                loss = train(model = model, graph = train_group, optimizer = optimizer, loss_fn = loss_fn )
                train_output = test(args=args, model=model, graph=train_group, loss_fn=loss_fn, epoch=epoch,  mode = 'Train')
                train_loss = train_output['loss'] 
                test_output = test(args = args, model = model, graph = test_group, loss_fn = loss_fn, epoch = epoch, mode = 'Test')
                test_loss = test_output['loss']
                current_epoch_f1 = test_output['F1']

                # If the current performance is better than the best performance so far, we update our model
                if ((current_epoch_f1 > best_each_group_dct['F1']) or (current_epoch_f1 == best_each_group_dct['F1'] and best_each_group_dct['AP'] < test_output['AP'])):
                    best_each_group_dct['AP'] = test_output['AP']
                    best_each_group_dct['P'] = test_output['P']
                    best_each_group_dct['R'] = test_output['R']
                    best_each_group_dct['F1'] = current_epoch_f1
                    best_each_group_dct['AUROC'] = test_output['AUROC']
                    best_each_group_dct['model'] = copy.deepcopy(model)
                    best_each_group_dct['model_loc'] = epoch
                    best_each_group_dct['optimizer_dict'] = copy.deepcopy(optimizer.state_dict())

                if epoch % args.num_epochs_print == 0:
                    print(f'Epoch: {epoch:03d}, Loss: {loss:.6f}, Train: {train_loss:.6f}, Test: {test_loss:.6f}, Time per epoch: {(time.time() - t_epoch):.4f}')
                # if epoch == args.num_epochs:
                #     if best_each_group_dct['model'] == None:
                #         print('WARNING: Current batch has ALL 0 F1 and AP (Model = None Type), pass current model to next batch')
                #         best_each_group_dct['model'] = copy.deepcopy(best_all_dct['model'][i])
                #         best_each_group_dct['optimizer_dict'] = copy.deepcopy(best_all_dct['optimizer_dict'][i]) 
                if epoch == args.num_epochs:   
                    if best_each_group_dct['model'] == None:
                        print('WARNING: Current group has ALL 0 F1 and AP (Model = None Type), pass current model to next group')
                        best_each_group_dct['model'] = copy.deepcopy(model)
                        best_each_group_dct['optimizer_dict'] = copy.deepcopy(optimizer.state_dict())
                        none_models = True
                        none_models_lst.append(True)
              #################### finish iteration for all epochs #################  
            #Passing the model and optimizer to the next group
            model = best_each_group_dct['model'].to(args.device)
            opt_state_dict = best_each_group_dct['optimizer_dict']
            optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
            if not none_models:
                optimizer.load_state_dict(opt_state_dict)
            
            ############################ Storing best results per group in big dictionary #########################
           
            best_all_dct['AP'].append(best_each_group_dct['AP'])
            best_all_dct['P'].append(best_each_group_dct['P'])
            best_all_dct['R'].append(best_each_group_dct['R'])
            best_all_dct['F1'].append(best_each_group_dct['F1'])
            # best_all_dct['model'].append(copy.deepcopy(best_each_group_dct['model']))
            # best_all_dct['optimizer_dict'].append(copy.deepcopy(best_each_group_dct['optimizer_dict']))
            best_all_dct['AUROC'].append(best_each_group_dct['AUROC'])
            best_all_dct['model_loc'].append(best_each_group_dct['model_loc'])

            # print time per batch
            print (f'Time per group: {(time.time()-t_batch):.4f}')

            # for last batch, save model
            if i == args.num_groups - 2:
                final_model = best_each_group_dct['model'].to(args.device)
                final_opt_dict = best_each_group_dct['optimizer_dict']
                optimizer = torch.optim.Adam(final_model.parameters(), lr=config["lr"], weight_decay=config['weight_decay'])
                if sum(none_models_lst) == 0:
                    optimizer.load_state_dict(final_opt_dict)
                saveModel(args, final_model,optimizer,best_all_dct['F1'][-1],best_all_dct['AP'][-1],best_all_dct['P'][-1],best_all_dct['R'][-1], args.modelPath)
        # print average results and time for the whole training
        print("Finished training!")
        # write best results per i in csv instead of write them all when all groups are finished!
        df = pd.DataFrame.from_dict({k:best_all_dct[k] for k in ('AP','P','R','F1','AUROC','model_loc') if k in best_all_dct})
        # df.to_csv(args.csvPath, index=False, header=True)
        avg_AP = df['AP'].mean()
        avg_F1 = df['F1'].mean()
        avg_AUROC = df['AUROC'].mean()
        avg_P = df['P'].mean()
        avg_R = df['R'].mean()

        avg_row = {
            'AP':avg_AP,
            'P':avg_P,
            'R':avg_R,
            'F1': avg_F1,
            'AUROC': avg_AUROC,
            'model_loc': df['model_loc'].mean()
        }        

        df2 = df.append(avg_row,ignore_index=True)

        param_row = {
            'AP':'alpha= ' + str(args.alpha),
            'P':'gamma= ' + str(args.gamma),
            'R':'learning_rate= ' + str(args.lr),
            'F1': 'hidden_channels=' + str(args.hidden_chnl),
            'AUROC': 'layers= ' + str(args.num_layers),
            'model_loc': 'dropout= ' + str(args.dropout),
        }        

        df2 = df2.append(param_row,ignore_index=True)

        df2.to_csv(args.csvPath, index=False, header=True)
        print(f'Average AP: {avg_AP:.4f}, Average F1: {avg_F1:.4f}, Average Precision: {avg_P:.4f},  Average AUROC: {avg_AUROC:.4f}, Average Recall: {avg_R:.4f}, Total Time: {(time.time() - t0):.4f} ')
    return avg_AP,avg_F1,avg_P,avg_R,avg_AUROC