import torch
import torch_geometric
import torch_geometric.data as data
import torch_geometric.transforms as T

def allnodes(G, undirected=False):
    #Accessing information of a PyG Graph
    ntype       = G.node_types
    etype       = G.edge_types[0]
    edge_idx    = G[etype].edge_index
    edge_lab    = G[etype].edge_label if hasattr(G[etype],'edge_label') else None
    edge_feature = G[etype].edge_attr if hasattr(G[etype],'edge_attr') else None
    edge_time = G[etype].edge_ts if hasattr(G[etype],'edge_ts') else None

    # Accessing information
    arange = torch.arange(edge_idx.size(1))
    stacked = torch.vstack((edge_idx[0],arange))
    big_nodes_mtrx = torch.vstack((stacked,edge_idx[1]))
    G_all = data.Data() if type(G) == torch_geometric.data.hetero_data.Data else data.HeteroData()

    for node in ntype:
        G_all[node].x = G[node].x
    G_all['trans_node'].x = edge_feature
    G_all['trans_node'].y = edge_lab
    G_all['trans_node'].ts = edge_time

    if hasattr(G[etype], "original_index"):
        G_all['trans_node'].original_index = G[etype].original_index
    if hasattr(G[etype], "edge_testgraph_mask"):
        G_all['trans_node'].testgraph_mask = G[etype].edge_testgraph_mask

    # G_all['trans_node'].mask = edge_mask
    G_all[ntype[0],'edge','trans_node'].edge_index = big_nodes_mtrx[:2]
    G_all['trans_node','edge',ntype[1]].edge_index = big_nodes_mtrx[1:]
    if undirected:
        G_all = T.ToUndirected()(G_all)
    return G_all