from copy import deepcopy
import pandas as pd
from torch_geometric.data import HeteroData 
import torch_geometric.transforms as T
import torch

def CG_Dataframe_Index(graph):
        data = deepcopy(graph) #Because ToUndirected will effect the orginal graph and that may be undesirable
        data = T.ToUndirected()(data)
        edge_df_list = []
        for cannoical_rel in data.metadata()[-1]:
            src, rel, dst = cannoical_rel
            edge_list = data[cannoical_rel]['edge_index'].tolist()
            edge_index_dict = {'src_node':edge_list[0], 'dst_node':edge_list[1]}
            for k, v in data[cannoical_rel].__dict__['_mapping'].items():
                if k!='edge_index':
                    edge_index_dict[k] = v.tolist()
            edge_index_df = pd.DataFrame(edge_index_dict)
            edge_index_df['edge_index'] = edge_index_df.index
            edge_index_df['src_node'] = edge_index_df['src_node'].apply(lambda x: f"{src}::{x}")
            edge_index_df['dst_node'] = edge_index_df['dst_node'].apply(lambda x: f"{dst}::{x}")
            edge_index_df['edge_index'] = edge_index_df['edge_index'].apply(lambda x: f"{rel}::{x}")
            if hasattr(data[cannoical_rel],'edge_label'):
                edge_index_df['edge_label'] = data[cannoical_rel]['edge_label'].tolist()
            edge_df_list.append(edge_index_df)
        edge_index_df = pd.concat(edge_df_list, ignore_index=True)
        merge_df = edge_index_df.merge(edge_index_df,left_on=['dst_node'],right_on=['src_node'],how = 'inner')
        merge_df.rename(columns={'edge_index_x':'src_node', 'edge_index_y':'dst_node'}, inplace=True)
        hyper_graph_edge_index = merge_df[["src_node","dst_node"]]
        return hyper_graph_edge_index
        
def CGT(data):
    """
    This functions takes a heterogenous graph and transforms it into a conjugate graph.
    This process can be RAM intensive and might fail with a graph will a large amount of edges

    Parameters:
    *data: A pyg HeteroData graph
    """
    pd.set_option('mode.chained_assignment', None) #To get Pandas to not complain about modifying view

    node_list = data.metadata()[0]
    source = node_list[0] 
    sink = node_list[1] 
    relationship = data.metadata()[-1][0][1] 
    #Initialize Graph
    hyper_data = HeteroData()
    if hasattr(data[source, relationship, sink], "edge_attr"):
        hyper_data[relationship].x = data[source, relationship, sink].edge_attr
    if hasattr(data[source, relationship, sink], "edge_ts"):
        hyper_data[relationship].ts = data[source, relationship, sink].edge_ts
    if hasattr(data[source, relationship, sink], "edge_label"):
        hyper_data[relationship].y = data[source, relationship, sink].edge_label
    if hasattr(data[source, relationship, sink], "original_index"):
        hyper_data[relationship].original_index = data[source, relationship, sink].original_index
    if hasattr(data[source, relationship, sink], "edge_testgraph_mask"):
        hyper_data[relationship].testgraph_mask = data[source, relationship, sink].edge_testgraph_mask 

    #Conjugate Graph Transform
    hyper_graph_edge_index= CG_Dataframe_Index(data)   
    
    #Working with action_df
    action_df=hyper_graph_edge_index[hyper_graph_edge_index['src_node'].str.startswith(relationship)]
    action_df[[relationship,'src_node']]=action_df['src_node'].str.split('::', expand=True)
    action_df[['rev_' + relationship,'dst_node']]=action_df['dst_node'].str.split('::', expand=True)
    action_df["src_node"] = pd.to_numeric(action_df["src_node"])
    action_df["dst_node"] = pd.to_numeric(action_df["dst_node"])

    hyper_data[relationship, sink, relationship].edge_index = torch.tensor([action_df['src_node'].tolist(),action_df['dst_node'].tolist()],dtype=torch.long)

    del action_df #To Save RAM

    rev_action_df=hyper_graph_edge_index[hyper_graph_edge_index['src_node'].str.startswith('rev_' + relationship)]
    rev_action_df[['rev_' + relationship,'src_node']]=rev_action_df['src_node'].str.split('::', expand=True)
    rev_action_df[[relationship,'dst_node']]=rev_action_df['dst_node'].str.split('::', expand=True)

    rev_action_df["src_node"] = pd.to_numeric(rev_action_df["src_node"])
    rev_action_df["dst_node"] = pd.to_numeric(rev_action_df["dst_node"])

    hyper_data[relationship, source, relationship].edge_index = torch.tensor([rev_action_df['src_node'].tolist(),rev_action_df['dst_node'].tolist()],dtype=torch.long)

    del rev_action_df #To Save RAM
    del hyper_graph_edge_index #To Save RAM

    # Create Sink Edge Features
    if hasattr(data[sink], "x"):
        hyper_sink = hyper_data[relationship, sink, relationship].edge_index[0] 
        sink_index = data[source, relationship, sink].edge_index[1] 
        sink_edge_index = sink_index[hyper_sink] 
        hyper_data[relationship, sink, relationship].edge_attr = data[sink].x[sink_edge_index,:]
        
    # Create Source Edge Features
    if hasattr(data[source], "x"):
        hyper_source = hyper_data[relationship, source, relationship].edge_index[0] 
        source_index = data[source, relationship, sink].edge_index[0] 
        source_edge_index = source_index[hyper_source] 
        hyper_data[relationship, source, relationship].edge_attr = data[source].x[source_edge_index,:]
    return hyper_data

