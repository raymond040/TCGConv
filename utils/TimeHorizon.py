from torch_geometric.utils import subgraph
import numpy as np
from copy import deepcopy
from tqdm import tqdm 

def Time_Groups(Graph, node_name = None, edge_temporal= False, num_groups = None, timelength = None, HeteroEdges = False, Test_Batch = True):
    """
    This function takes a heterogenous graph and returns a series of temporal subsets
    based on a single nodetype's temporal value or single edge_type temporal value 
    
    Parameters:
    * Graph: A pyg heterogenous graph
    * node_name (str): The name of node which has temporal information (e.g. 'action')
    * Edge_temporal (bool): When edges are temporal instead of nodes. Node name is not needed in this case
    * num_groups (int): The amount of groups the graph will be dvided into. Uses percentile to make subsets of graphs
    * timelengh (float): For fixed length subsets, value has to be in the orginal graph value. For example graph is seconds then timelengh would need to be in secs
    * HeteroEdges (bool): When your heterograph has a mix of node types in the edges e.g ('action' 'edge' 'class') instead of ('action', 'edge', 'action')
    
    Returns: 2 list of HeteroGraphs one for G_i and the other for (G_i and G_i+1) combined
    """
    edge_relationships= Graph.metadata()[-1]
    if edge_temporal == True:
        time_series  = Graph[edge_relationships[0]].edge_ts
    elif hasattr(Graph[node_name],'ts'):
        time_series = Graph[node_name].ts
    else:
        raise Exception("Could not find time series in graph. Make sure attribute is ts or edge_ts")
    
    if num_groups is not None:
        percentiles = np.linspace(0,1,num = num_groups + 1)
        batch_values = np.quantile(time_series, percentiles) # Will include max value
        
    elif timelength is not None:
        batch_values = np.arange(0, max(time_series), step = timelength) # Will not always include max value
        batch_values = np.append(batch_values, max(time_series)) # append Max value to include those in groups
    else:
        raise Exception("Please set percentile groups or timelength")       
    
    # Creates tuples of batch value and the next batch value
    batch_ranges_training = list(map(lambda x: (batch_values[x], batch_values[x+1]), range(0,len(batch_values)-1)))
    batch_ranges_testing = list(map(lambda x: (batch_values[x], batch_values[x+2]), range(0,len(batch_values)-2)))
    all_batch_ranges = [batch_ranges_training, batch_ranges_testing]
    Train_Test_Graph_Groups = []
    testing = 0
    for batch_range in all_batch_ranges:
        Graph_Groups = []
        testing += 1 # When testing = 2 then we are building the testing graphs
        if testing == 2 and Test_Batch == False: # if Test_Batch is False then will skip the testing groups
            continue
        for i, batch in tqdm(enumerate(batch_range)):
            start, stop = batch
            dummy_graph = deepcopy(Graph) # Creates a copy of the graph to change it's index
            # for cornor case
            if stop == time_series[time_series.shape[0]-1]:
                batch_mask = (time_series >= start) & (time_series <= stop)
            else:
                batch_mask = (time_series >= start) & (time_series < stop)
            if sum(batch_mask) == 0: # Don't include subgraphs that have no nodes
                continue
            batch_index = np.where(batch_mask)
            if edge_temporal == True:
                # Transfer Attributes from orginal graphs the edges
                relationship = edge_relationships[0]
                if hasattr(Graph[relationship],'edge_index'):
                    dummy_graph[relationship].edge_index = Graph[relationship].edge_index[:,batch_index[0]]
                if hasattr(Graph[relationship],'edge_attr'): #Harder
                    dummy_graph[relationship].edge_attr = Graph[relationship].edge_attr[batch_index[0],:]
                if hasattr(Graph[relationship],'edge_label'):
                    dummy_graph[relationship].edge_label = Graph[relationship].edge_label[batch_index[0]]
                if hasattr(Graph[relationship],'edge_ts'):
                    dummy_graph[relationship].edge_ts = Graph[relationship].edge_ts[batch_index[0]]
                if hasattr(Graph[relationship],'edge_mask'):
                    dummy_graph[relationship].edge_mask = Graph[relationship].edge_mask[batch_index[0]]
                dummy_graph[relationship].original_index = batch_index[0]
                if testing == 2 and Test_Batch == True: #If creating testing graph give mask for the previous subgraph
                    start_sub, stop_sub = batch_ranges_training[i]
                    time_series_sub = dummy_graph[relationship].edge_ts
                    if stop == time_series_sub[time_series_sub.shape[0]-1]:
                        sub_batch_mask = (time_series_sub >= start_sub) & (time_series_sub <= stop_sub)
                    else:
                        sub_batch_mask = (time_series_sub >= start_sub) & (time_series_sub < stop_sub)
                    
                    dummy_graph[relationship].edge_testgraph_mask = ~sub_batch_mask #~ to make test graph true
                    
                Graph_Groups.append(dummy_graph)
                continue 

            # Transfer Node Attributes from orginal graph
            dummy_graph[node_name].original_index = batch_index[0]
            if hasattr(Graph[node_name],'x'):
                dummy_graph[node_name].x = Graph[node_name].x[batch_index[0]]
            if hasattr(Graph[node_name],'ts'):
                dummy_graph[node_name].ts = Graph[node_name].ts[batch_index[0]]
            if hasattr(Graph[node_name],'mask'):
                dummy_graph[node_name].mask = Graph[node_name].mask[batch_index[0]]
            if hasattr(Graph[node_name],'y'):
                dummy_graph[node_name].y = Graph[node_name].y[batch_index[0]]
            
            if testing == 2: # If creating testing graph give mask for the previous subgraph
                    start_sub, stop_sub = batch_ranges_training[i]
                    time_series_sub = dummy_graph[node_name].ts
                    if stop == time_series_sub[time_series_sub.shape[0]-1]:
                        sub_batch_mask = (time_series_sub >= start_sub) & (time_series_sub <= stop_sub)
                    else:
                        sub_batch_mask = (time_series_sub >= start_sub) & (time_series_sub < stop_sub)
                    
                    dummy_graph[node_name].testgraph_mask = ~sub_batch_mask #~ to make test graph true
                    

            # For each realtionship crate subgraph of edge_index
            if HeteroEdges == True: # Because edges have different ids need to filter on only the one specified
                for relationship in edge_relationships:
                    edge_index_of_interest  = np.where(list(map(lambda x: x == node_name, relationship)))[0][0] #Where is trans_node in the realtionship
                    if edge_index_of_interest == 2: # Since edge index only has 2 rows need index to be = to 1
                        edge_index_of_interest = 1 
                    Original_Nodes = Graph[relationship].edge_index[edge_index_of_interest]
                    subedge_index = np.where(np.isin(Original_Nodes, batch_index))[0]
                    subedge = Graph[relationship].edge_index[:,subedge_index]
                    subedge[edge_index_of_interest]= subedge[edge_index_of_interest] - min(batch_index[0]) #Setting namednode's edges to zero
                    dummy_graph[relationship].edge_index = subedge
            elif HeteroEdges == False:
                for relationship in edge_relationships:
                    subedge = subgraph(batch_index, Graph[relationship].edge_index)[0]
                    dummy_graph[relationship].edge_index = subedge - min(batch_index[0]) #Setting namednode's edges to zero

            else:
                raise Exception("HeteroEdges must be True or False") 

            Graph_Groups.append(dummy_graph)
        Train_Test_Graph_Groups.append(Graph_Groups)
    if Test_Batch == True:
        return Train_Test_Graph_Groups[0], Train_Test_Graph_Groups[1] #[0] is (G_i), [1] is (G_i, G_i+1) combined
    elif Test_Batch == False:
        return Train_Test_Graph_Groups[0] # [0] is (G_i)