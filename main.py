import os
import sys
import copy
import torch_geometric.transforms as T

sys.path.insert(0, '/workspaces/Edge-Representation-Learning-in-Temporal-Graph/utils')
from util import setup_device,setup_seed,clean
from config import parse_args
from DatasetPrep import DatasetPrep
from TimeHorizon import Time_Groups
from CGT import CGT
from ATN import allnodes
sys.path.insert(0, '/workspaces/Edge-Representation-Learning-in-Temporal-Graph/Models')
from Model_CONCAT import CONCAT_Trainer
from Model_ATN import ATN_Trainer
from Model_TCGConv import TCGConv_Trainer
from Model_TCGConv_sum import TCGConv_sum_Trainer
from Model_CGConv import CGConv_Trainer
from Model_CGConv_sum import CGConv_sum_Trainer


def main():
    clean()
    args= parse_args()
    setup_seed(args)
    setup_device(args)
    args.csvPath = os.path.join(args.root, 'Results/'+args.dataset_name+'/'+args.model_type+'_'+args.dataset_name+'_'+args.type_ED+'_'+str(args.num_version)+'.csv')
    args.modelPath = os.path.join(args.root, 'Results/'+args.dataset_name+'/'+args.model_type+'_'+args.dataset_name+'_'+args.type_ED+'_'+str(args.num_version)+'.pth')

    config = {
            "alpha": args.alpha,
            "gamma": args.gamma,
            "lr": args.lr,
            "hidden_chnl": args.hidden_chnl,
            "dropout": args.dropout,
            "num_layers": args.num_layers,
            "weight_decay": args.weight_decay,
        }
    
    if args.dataset_name =="CC":
        args.root_readData = args.root+"Dataset/Credit_Card_Fraud"
    elif args.dataset_name == "MOOC":
        args.root_readData = args.root+"Dataset/MOOC"
    else:
        raise Exception("Unknown dataset_name")

    # data preparation
    Graph = DatasetPrep(dataset_name = args.dataset_name, root = args.root_readData, type = args.graph_type, percentage = args.percentage, truncate_size= args.truncate_size)[0]
    if args.model_type == "CONCAT":
        G_Groups, G_Test = Time_Groups(Graph, edge_temporal = True, num_groups = args.num_groups, Test_Batch = True)

        if args.type_ED == "TT":
            CONCAT_Trainer(args, config, G_Groups, G_Test)
        elif args.type_ED == "sub":
            CONCAT_Trainer(args, config, G_Groups, None)
        else:
            raise Exception("type_ED is invalid, only sub and TT are supported.")

    elif args.model_type == "ATN":
        Graph = allnodes(Graph)
        Graph = T.ToUndirected()(Graph)
        G_Groups, G_Test = Time_Groups(Graph, 'trans_node', num_groups = args.num_groups, HeteroEdges = True, Test_Batch = True)

        if args.type_ED == "TT":
            ATN_Trainer(args, config, G_Groups, G_Test)
        elif args.type_ED == "sub":
            ATN_Trainer(args, config, G_Groups, None)
        else:
            raise Exception("type_ED is invalid, only sub and TT are supported (case sensitive).")

    elif args.model_type == "TCGConv":
        G_Groups,G_Test = Time_Groups(Graph, edge_temporal = True, num_groups = args.num_groups, HeteroEdges = False, Test_Batch = True)
        CG_Groups = map(CGT, G_Groups)
        CG_Test = map(CGT, G_Test)

        if args.type_ED == "TT":
            TCGConv_Trainer(args, config, CG_Groups, CG_Test)
        elif args.type_ED == "sub":
            TCGConv_Trainer(args, config, CG_Groups, copy.deepcopy(CG_Groups))
        else:
            raise Exception("type_ED is invalid, only sub and TT are supported (case sensitive).")

    elif args.model_type == "TCGConv_sum":
        G_Groups,G_Test = Time_Groups(Graph, edge_temporal = True, num_groups = args.num_groups, HeteroEdges = False, Test_Batch = True)
        CG_Groups = map(CGT, G_Groups)
        CG_Test = map(CGT, G_Test)

        if args.type_ED == "TT":
            TCGConv_sum_Trainer(args, config, CG_Groups, CG_Test)
        elif args.type_ED == "sub":
            TCGConv_sum_Trainer(args, config, CG_Groups, copy.deepcopy(CG_Groups))
        else:
            raise Exception("type_ED is invalid, only sub and TT are supported (case sensitive).")

    elif args.model_type == "CGConv":
        G_Groups,G_Test = Time_Groups(Graph, edge_temporal = True, num_groups = args.num_groups, HeteroEdges = False, Test_Batch = True)
        CG_Groups = map(CGT, G_Groups)
        CG_Test = map(CGT, G_Test)

        if args.type_ED == "TT":
            CGConv_Trainer(args, config, CG_Groups, CG_Test)
        elif args.type_ED == "sub":
            CGConv_Trainer(args, config, CG_Groups, copy.deepcopy(CG_Groups))
        else:
            raise Exception("type_ED is invalid, only sub and TT are supported (case sensitive).")
    
    elif args.model_type == "CGConv_sum":
        G_Groups,G_Test = Time_Groups(Graph, edge_temporal = True, num_groups = args.num_groups, HeteroEdges = False, Test_Batch = True)
        CG_Groups = map(CGT, G_Groups)
        CG_Test = map(CGT, G_Test)

        if args.type_ED == "TT":
            CGConv_sum_Trainer(args, config, CG_Groups, CG_Test)
        elif args.type_ED == "sub":
            CGConv_sum_Trainer(args, config, CG_Groups, copy.deepcopy(CG_Groups))
        else:
            raise Exception("type_ED is invalid, only sub and TT are supported (case sensitive).")

    else:
        raise Exception("model_type is invalid, please choose among CONCAT, ATN, TCGConv, TCGConv_sum, CGConv, CGConv_sum (case sensitive).")


if __name__ == '__main__':
    main()