import os
import sys
import copy
from numpy import true_divide
import torch_geometric.transforms as T
from numpy.random import uniform
from numpy.random import choice
from scipy.stats import loguniform
import pandas as pd

HPC_Flag = True
if HPC_Flag:
    sys.path.insert(0, '/home/svu/e0407728/My_FYP/TCGConv/utils')
else:
    sys.path.insert(0, '/workspaces/TCGConv/utils')
from util import setup_device,setup_seed,clean
from config import parse_args
from DatasetPrep import DatasetPrep
from TimeHorizon import Time_Groups
from CGT import CGT
from ATN import allnodes
if HPC_Flag:
    sys.path.insert(0, '/home/svu/e0407728/My_FYP/TCGConv/Models')
else:
    sys.path.insert(0, '/workspaces/TCGConv/Models')
from Model_CONCAT import CONCAT_Trainer
from Model_ATN import ATN_Trainer
from Model_TCGConv import TCGConv_Trainer
from Model_TCGConv_sum import TCGConv_sum_Trainer
from Model_CGConv import CGConv_Trainer
from Model_CGConv_sum import CGConv_sum_Trainer


def main():
    clean()
    args= parse_args()
    #setup_seed(args)
    setup_device(args)
    args.csvPath = os.path.join(args.root, 'Results/'+args.dataset_name+'/'+args.model_type+'_'+args.dataset_name+'_'+args.type_ED+'_'+str(args.num_version)+'.csv')
    args.modelPath = os.path.join(args.root, 'Results/'+args.dataset_name+'/'+args.model_type+'_'+args.dataset_name+'_'+args.type_ED+'_'+str(args.num_version)+'.pth')

    config = {
            "alpha": uniform(0,0.05),
            "gamma": choice([2,3]),
            "lr": loguniform.rvs(1e-8,1e-2).item(),
            "hidden_chnl": choice([32,64]),
            "dropout": uniform(0,1),
            "num_layers":choice([2,3]),
            "weight_decay": loguniform.rvs(1e-8,1e-4).item(),
        }
        
    if args.dataset_name =="CC":
        args.root_readData = args.root+"Dataset/Credit_Card_Fraud"
    elif args.dataset_name == "MOOC":
        args.root_readData = args.root+"Dataset/MOOC"
    elif args.dataset_name == "W":
        args.root_readData = args.root+"Dataset/Wikipedia"
    elif args.dataset_name == "R":
        args.root_readData = args.root+"Dataset/Reddit"
    else:
        raise Exception("Unknown dataset_name")

    # data preparation
    Graph = DatasetPrep(dataset_name = args.dataset_name, root = args.root_readData, type = args.graph_type, percentage = args.percentage, truncate_size= args.truncate_size)[0]
    if args.model_type == "CONCAT":
        G_Groups, G_Test = Time_Groups(Graph, edge_temporal = True, num_groups = args.num_groups, Test_Batch = True)

        if args.type_ED == "TT":
            results = CONCAT_Trainer(args, config, G_Groups, G_Test)
        elif args.type_ED == "sub":
            results = CONCAT_Trainer(args, config, G_Groups, None)
        else:
            raise Exception("type_ED is invalid, only sub and TT are supported.")

    elif args.model_type == "ATN":
        Graph = allnodes(Graph)
        Graph = T.ToUndirected()(Graph)
        G_Groups, G_Test = Time_Groups(Graph, 'trans_node', num_groups = args.num_groups, HeteroEdges = True, Test_Batch = True)

        if args.type_ED == "TT":
            results = ATN_Trainer(args, config, G_Groups, G_Test)
        elif args.type_ED == "sub":
            results = ATN_Trainer(args, config, G_Groups, None)
        else:
            raise Exception("type_ED is invalid, only sub and TT are supported (case sensitive).")

    elif args.model_type == "TCGConv":
        G_Groups,G_Test = Time_Groups(Graph, edge_temporal = True, num_groups = args.num_groups, HeteroEdges = False, Test_Batch = True)
        CG_Groups = map(CGT, G_Groups)
        CG_Test = map(CGT, G_Test)

        if args.type_ED == "TT":
            results = TCGConv_Trainer(args, config, CG_Groups, CG_Test)
        elif args.type_ED == "sub":
            results = TCGConv_Trainer(args, config, CG_Groups, copy.deepcopy(CG_Groups))
        else:
            raise Exception("type_ED is invalid, only sub and TT are supported (case sensitive).")

    elif args.model_type == "TCGConv_sum":
        G_Groups,G_Test = Time_Groups(Graph, edge_temporal = True, num_groups = args.num_groups, HeteroEdges = False, Test_Batch = True)
        CG_Groups = map(CGT, G_Groups)
        CG_Test = map(CGT, G_Test)

        if args.type_ED == "TT":
            results = TCGConv_sum_Trainer(args, config, CG_Groups, CG_Test)
        elif args.type_ED == "sub":
            results = TCGConv_sum_Trainer(args, config, CG_Groups, copy.deepcopy(CG_Groups))
        else:
            raise Exception("type_ED is invalid, only sub and TT are supported (case sensitive).")

    elif args.model_type == "CGConv":
        G_Groups,G_Test = Time_Groups(Graph, edge_temporal = True, num_groups = args.num_groups, HeteroEdges = False, Test_Batch = True)
        CG_Groups = map(CGT, G_Groups)
        CG_Test = map(CGT, G_Test)

        if args.type_ED == "TT":
            results = CGConv_Trainer(args, config, CG_Groups, CG_Test)
        elif args.type_ED == "sub":
            results = CGConv_Trainer(args, config, CG_Groups, copy.deepcopy(CG_Groups))
        else:
            raise Exception("type_ED is invalid, only sub and TT are supported (case sensitive).")
    
    elif args.model_type == "CGConv_sum":
        G_Groups,G_Test = Time_Groups(Graph, edge_temporal = True, num_groups = args.num_groups, HeteroEdges = False, Test_Batch = True)
        CG_Groups = map(CGT, G_Groups)
        CG_Test = map(CGT, G_Test)

        if args.type_ED == "TT":
            results = CGConv_sum_Trainer(args, config, CG_Groups, CG_Test)
        elif args.type_ED == "sub":
            results = CGConv_sum_Trainer(args, config, CG_Groups, copy.deepcopy(CG_Groups))
        else:
            raise Exception("type_ED is invalid, only sub and TT are supported (case sensitive).")

    else:
        raise Exception("model_type is invalid, please choose among CONCAT, ATN, TCGConv, TCGConv_sum, CGConv, CGConv_sum (case sensitive).")
    
    print(results)
    avgAP = results[0]
    avgF1 = results[1]
    print(config)
    big_dct['avg_F1'].append(avgF1)
    big_dct['avg_AP'].append(avgAP)
    big_dct['alpha'].append(config['alpha'])
    big_dct['gamma'].append(config['gamma'])
    big_dct['lr'].append(config['lr'])
    big_dct['hidden_chnl'].append(config['hidden_chnl'])
    big_dct['dropout'].append(config['dropout'])
    big_dct['num_layers'].append(config['num_layers'])
    big_dct['weight_decay'].append(config['weight_decay'])



if __name__ == '__main__':
    args = parse_args()
    #setup_device(args)
    HPath = os.path.join(args.root, 'Tune/'+args.dataset_name+'/H_'+args.model_type+'_'+str(args.num_version)+'.csv')
    big_dct = {
            'avg_F1': [],
            'avg_AP': [],
            'alpha': [],
            'gamma': [],
            'lr': [],
            'weight_decay': [],
            'hidden_chnl':  [],
            'dropout': [],
            'num_layers': []
        }
    for i in range (0, args.n_run):
        main()
        #setup_seed(args)
        big_df = pd.DataFrame.from_dict(big_dct)
        big_df.to_csv(HPath, index=False, header=True)