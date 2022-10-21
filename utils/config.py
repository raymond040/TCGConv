import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Model parameters")

    parser.add_argument("--seed", type=str, default='2022', help="random seed.")
    parser.add_argument("--mode", type=str, default='train', help="whether you are training a new model or using pretrained models, choose between train and test")
    parser.add_argument('--model_type', type=str, default="TCGConv", help="choose among CONCAT, ATN, TCGConv, TCGConv_sum, CGConv, CGConv_sum")
    # ========================= Data Configs ==========================
    parser.add_argument('--root', type=str, default='/workspaces/Edge-Representation-Learning-in-Temporal-Graph/')
    parser.add_argument('--graph_type', type=str, default='G', help = "Please use default graph type")
    parser.add_argument('--dataset_name', type=str, default='CC') 
    parser.add_argument('--percentage', type=float, default=1, help='percent of data used')
    parser.add_argument('--truncate_size', type=int, default=None)
    parser.add_argument('--k', type=int, default=20, help='number of k for top k')

    # ======================== Experiment Design =========================
    parser.add_argument('--type_ED', type=str, default='sub', help='type of time horizon method, choose between TT and sub')

    # ======================== SavedModel Configs =========================
    parser.add_argument('--num_version', type=str, default='9999')

    # ========================= Training Configs ==========================
    parser.add_argument('--num_groups', type=int, default=100, help='How many groups? usually 100 for CC and 30 for MOOC')
    parser.add_argument('--num_epochs', type=int, default=250, help='How many epochs')
    parser.add_argument('--num_epochs_print', type=int, default=50, help='How many epochs to print the results')
    parser.add_argument('--weight', default=None, help='weight initialization')

    parser.add_argument("--alpha", default=0.01, type=float, help="alpha for focal loss; class weight")
    parser.add_argument("--gamma", default=2, type=float, help="gamma for focal loss; used to adjust the focus on difficult and easy samples, it is set as 2 in retainnet.")
    parser.add_argument('--lr', default=1e-4, type=float, help='initial learning rate')
    parser.add_argument("--weight_decay", default=1e-8, type=float, help="Weight decay if we apply some.")
    
    # ========================== Modeling =============================
    parser.add_argument('--hidden_chnl', type=int, default=64) 
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout ratio')
    parser.add_argument('--num_layers', type=int, default=3, help='number of GCN layers')

    # ========================== HyperTune Parameters =============================
    # parser.add_argument('--num_cpu_per_trial', type=int, default=2, help='number of CPUs per trial')
    # parser.add_argument('--num_gpu_per_trial', type=int, default=0, help='number of GPUs per trial')
    # parser.add_argument('--metric', type=str, default="avg_F1", help='choose between avg_F1 and avg_AP')
    # parser.add_argument('--mode', type=str, default="max")
    # parser.add_argument('--num_samples', type=int, default=2, help='number of combinations of config to try')
    # parser.add_argument('--max_t', type=int, default=1)
    # parser.add_argument('--grace_period', type=int, default=1)
    # parser.add_argument('--reduction_factor', type=int, default=2)
    
    

    return parser.parse_args()
