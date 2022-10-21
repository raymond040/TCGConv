#%%
import sys
HPCFlag=True

if HPCFlag==False:
    sys.path.insert(0, '/workspaces/Edge-Representation-Learning-in-Temporal-Graph/utils')
else:
    sys.path.insert(0, '/home/svu/e0407728/My_FYP/GNN/Edge-Representation-Learning-in-Temporal-Graph/utils/')
from Reading_Saving_Batches import Read_Zip
import numpy as np
import torch

def RecentK_Transform(graph, k = 20):
    """
    Trims edges based on the k recent edges. Meant to be used with conjugate graph.
    It sorts the edge index then finds the most recent edge and takes k steps back and slices. This happens for all unique nodes.
    If the node has less then k recent edges then it keeps them all.
    It does top K for all edge types
    It does not handle edge_attr yet
    Assumes node index has the temporal information
    """
    def Valid_Edges_mask(edgeindex):
    #Temporally speaking valid edges. Assumes edge index indicates time order
        from_node = edgeindex[0]
        to_node = edgeindex[1]
        return to_node >= from_node
    edge_types = graph.metadata()[-1]
    for edge in edge_types:
        edge_index = graph[edge].edge_index
        edge_mask = Valid_Edges_mask(edge_index)
        edge_index = edge_index[:,edge_mask]
        #Sorts edges very important to determine the most recent edge
        edge_index_sorted = edge_index[:,np.lexsort((edge_index[0], edge_index[1]))] #Sort by from node then to node
        #np.unique gives index of first apperance of index
        unique_values, first_unique_index = np.unique(edge_index_sorted[1], return_index= True)
        #So the difference indicates how many observations
        unique_diff = np.append(np.diff(first_unique_index) - 1, len(edge_index_sorted[1]) - first_unique_index[-1] - 1) # -1 since we have zero index
        last_unique_index = first_unique_index + unique_diff + 1 #add 1 to include most recent edge(which is the self loop)
        k_recent_index = np.where(k < (last_unique_index - first_unique_index), last_unique_index - k, first_unique_index) #If edges have more then k edges subtract k from the last unique index
        trimmed_edges = []
        for i in range(0, len(k_recent_index)):
            trimmed_edges.append(edge_index_sorted[:,np.s_[k_recent_index[i]:last_unique_index[i]]])
        New_Edges = torch.stack((torch.cat([x[0] for x in trimmed_edges]), torch.cat([x[1] for x in trimmed_edges])))
        graph[edge].edge_index = New_Edges
    return graph

# #%%
# MOOC_Sub, MOOC_Test = Read_Zip("../SubGraphs/MOOC_LineGraph_60.zip")
# MOOC_Sub = list(MOOC_Sub)
# RecentK_Transform(MOOC_Sub[0], 10)
# #%% Sanity Check
# edge_index = MOOC_Sub[0]["action", "activity", "action"].edge_index
# max(np.unique(edge_index[1], return_counts= True)[1]) #Should be equal to k
# ##
# edge_index = MOOC_Sub[0]["action", "user", "action"].edge_index
# max(np.unique(edge_index[1], return_counts= True)[1]) #Should be equal to k
