from os.path import join
from pandas import read_csv 
import pandas as pd
import torch 
from torch_geometric.data import HeteroData 
import torch_geometric.transforms as T
from datetime import date, datetime, timedelta
from torch.utils.data import Dataset
import os.path as osp
import os
import warnings
warnings.filterwarnings('ignore')

class DatasetPrep(Dataset):
    """
    This function reads in the MOOC/Credit_Card (CC) tsv's, truncates and outputs a HeteroData PyG graph dataset.
    Call this function by  `dataset = DatasetPrep(dataset_name, root, type, num_slice, days or percentage)`
    
    Parameters:
    * dataset_name (str): "MOOC" or "CC"
                    MOOC dataset contains ~30 days of user actions.
                    CC Dataset contains ~two years of transactions.
    * root (str): string with filepath where the mooc dataset is stored in your computer
    * type (str):   G - simply the graph
                    CG - Conjugate_graph approach: edge_ts information is in ts for the nodes. 
    * num_slice (int): number of slices to split the graph, The function splits the time range evenly. 
                    The full graph is stored in dataset[len(dataset)-1]
    * days (int) = None: The first number of days starting from the minimum time to keep in dataset
    * percentage (float) = None: The percentage of rows to keep in dataset starting from first row. (0. to 1.)
    * truncate_size (int) : The number of rows to keep in dataset starting from the first row
    Returns: Dataset
    Each graph can be accessed by `dataset[0]`
    """
    def __init__(self, dataset_name:str, root, type,num_slice = None, days = None, percentage = None, truncate_size = None):

        assert (days is None) or (percentage is None), "Please provide either days or percentage, but not both."
        if days is None and percentage is None:
            percentage = 1

        # read datafram and truncate
        if dataset_name == "MOOC":
            data = self.MOOC_HeteroGraph(root=root, days=days, percentage=percentage, truncate_size = truncate_size)

        elif dataset_name == "CC":
            data = self.Credit_Card_HeteroGraph(root=root, days=days, percentage=percentage, truncate_size = truncate_size)
        else:
            raise Exception("Dataset_name is invalid, only MOOC and CC are supported.")

        node_list = data.metadata()[0]
        source = node_list[0] 
        sink = node_list[1] 
        relationship = data.metadata()[-1][0][1] 
        
        # wrap graphs into dataset
        self.samples=[]
        if type == "G":
            # mask is to indicate val and test edges after splitting.                
            mask = torch.ones(data[source, relationship, sink].edge_index.shape[1], dtype=torch.bool) #Placeholder Mask 
            data[source, relationship, sink].edge_mask = mask
            self.samples.append(data)
                
        elif type == "CG":
            assert num_slice is not None, "num_slice is not None expected, got None."
            self.samples.append(data)
            hyper_graph_edge_index = self.ToHypergraph()

            # prepare edge_index for two relationships
            action_df=hyper_graph_edge_index[hyper_graph_edge_index['src_node'].str.startswith(relationship)]
            action_df[[relationship,'src_node']]=action_df['src_node'].str.split('::', expand=True)
            action_df[['rev_' + relationship,'dst_node']]=action_df['dst_node'].str.split('::', expand=True)
            rev_action_df=hyper_graph_edge_index[hyper_graph_edge_index['src_node'].str.startswith('rev_' + relationship)]
            rev_action_df[['rev_' + relationship,'src_node']]=rev_action_df['src_node'].str.split('::', expand=True)
            rev_action_df[[relationship,'dst_node']]=rev_action_df['dst_node'].str.split('::', expand=True)

            # convert datatype to number
            action_df["src_node"] = pd.to_numeric(action_df["src_node"])
            action_df["dst_node"] = pd.to_numeric(action_df["dst_node"])
            rev_action_df["src_node"] = pd.to_numeric(rev_action_df["src_node"])
            rev_action_df["dst_node"] = pd.to_numeric(rev_action_df["dst_node"])

            # initialize hypergraph
            hyper_data = HeteroData()
            hyper_data[relationship].x = data[source, relationship, sink].edge_attr
            hyper_data[relationship].ts = data[source, relationship, sink].edge_ts
            hyper_data[relationship].y = data[source, relationship, sink].edge_label
            hyper_data[relationship, sink, relationship].edge_index = torch.tensor([action_df['src_node'].tolist(),action_df['dst_node'].tolist()],dtype=torch.long)
            hyper_data[relationship, source, relationship].edge_index = torch.tensor([rev_action_df['src_node'].tolist(),rev_action_df['dst_node'].tolist()],dtype=torch.long)
            
            if dataset_name=="CC":
                # Create Sink Edge Features
                hyper_sink = hyper_data[relationship, sink, relationship].edge_index[0] 
                sink_index = data[source, relationship, sink].edge_index[1] 
                sink_edge_index = sink_index[hyper_sink] 
                hyper_data[relationship, sink, relationship].edge_attr = data[sink].x[sink_edge_index,:]
                
                # Create Source Edge Features
                hyper_source = hyper_data[relationship, source, relationship].edge_index[0] 
                source_index = data[source, relationship, sink].edge_index[0] 
                source_edge_index = source_index[hyper_source] 
                hyper_data[relationship, source, relationship].edge_attr = data[source].x[source_edge_index,:]

            assert (num_slice <= len(hyper_data[relationship].ts.unique())), "num_slice must be less than or equal to unique timestamp in the graph."
            # split the data into num_slice parts
            self.samples=[] # clean the list first
            for i in range(num_slice):
                if i == num_slice - 1:
                    split = len(hyper_data[relationship].ts.unique())-1
                else:
                    split = len(hyper_data[relationship].ts.unique())/num_slice
                    split = int((i+1) * split)-1
                split = hyper_data[relationship].ts.unique()[split].item() 

                index = (hyper_data[relationship].ts <= split).nonzero().squeeze()
                subset_dict = {relationship: index}
                subdata = hyper_data.subgraph(subset_dict)

                # we store all nodes of actions, while edges are sliced per timestamp
                subdata[relationship].x = hyper_data[relationship].x
                subdata[relationship].ts = hyper_data[relationship].ts
                subdata[relationship].y = hyper_data[relationship].y
                index_ = (hyper_data[relationship].ts > split).nonzero().squeeze()
                mask = torch.ones(hyper_data[relationship].x.shape[0], dtype=torch.bool)
                mask[index_] = False
                subdata[relationship].mask = mask
                self.samples.append(subdata)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def ToHypergraph(self):
        data = self.samples[0]
        data = T.ToUndirected()(data)
        edge_df_list = []
        for cannoical_rel in data.metadata()[-1]:
            src, rel, dst = cannoical_rel
            edge_list = data[cannoical_rel]['edge_index'].numpy().tolist()
            edge_index_dict = {'src_node':edge_list[0], 'dst_node':edge_list[1]}
            for k, v in data[cannoical_rel].__dict__['_mapping'].items():
                if k!='edge_index':
                    edge_index_dict[k] = v.numpy().tolist()
            edge_index_df = pd.DataFrame(edge_index_dict)
            edge_index_df['edge_index'] = edge_index_df.index
            edge_index_df['src_node'] = edge_index_df['src_node'].apply(lambda x: f"{src}::{x}")
            edge_index_df['dst_node'] = edge_index_df['dst_node'].apply(lambda x: f"{dst}::{x}")
            edge_index_df['edge_index'] = edge_index_df['edge_index'].apply(lambda x: f"{rel}::{x}")
            if hasattr(data[cannoical_rel],'edge_label'):
                edge_index_df['edge_label'] = data[cannoical_rel]['edge_label'].numpy().tolist()
            edge_df_list.append(edge_index_df)
        edge_index_df = pd.concat(edge_df_list, ignore_index=True)
        merge_df = edge_index_df.merge(edge_index_df,left_on=['dst_node'],right_on=['src_node'],how = 'inner')
        merge_df.rename(columns={'edge_index_x':'src_node', 'edge_index_y':'dst_node'}, inplace=True)
        hyper_graph_edge_index = merge_df[["src_node","dst_node"]]
        return hyper_graph_edge_index

    def MOOC_HeteroGraph(self, root, days, percentage, truncate_size):
        # read MOOC data
        actions_df = read_csv(join(root,"mooc_actions.tsv"),sep='\t')
        action_features_df = read_csv(join(root,"mooc_action_features.tsv"),sep='\t')
        action_labels_df = read_csv(join(root,"mooc_action_labels.tsv"),sep='\t')
        graph_data = actions_df.merge(action_features_df, on="ACTIONID", how="left").merge(action_labels_df, on="ACTIONID", how="left")

        # MOOC has na labels and duplicates
        graph_data = graph_data.dropna()
        graph_data = graph_data.drop_duplicates()

        if days is not None:
            assert days <= 30 and days >=0, "days out of range [0,30]."
            Min_Time = min(graph_data["TIMESTAMP"])
            Cut_Time = Min_Time + days * 86400 # Days times seconds/day
            graph_data = graph_data.loc[graph_data["TIMESTAMP"] <= Cut_Time,:]
            
        elif percentage is not None:
            assert percentage<=1 and percentage >=0, "percentage out of range [0,1]." 
            graph_data = graph_data.iloc[:round(graph_data.shape[0] * percentage),:]
        if truncate_size is not None:
            graph_data = graph_data.iloc[:truncate_size]
        # initialize HeteroData and create graph object
        data = HeteroData()

        # create graph objects
        data['user'].x = torch.arange(0, max(actions_df['USERID'])+1,dtype=torch.float32).reshape(-1,1)
        data['activity'].x = torch.arange(0, max(actions_df['TARGETID'])+1,dtype=torch.float32).reshape(-1,1)
        data['user', 'action', 'activity'].edge_index = torch.tensor([graph_data['USERID'].tolist(),graph_data['TARGETID'].tolist()],dtype=torch.long)
        data['user', 'action', 'activity'].edge_label = torch.tensor(graph_data['LABEL'].tolist(),dtype=torch.long)
        data['user', 'action', 'activity'].edge_attr = torch.tensor(graph_data[['FEATURE0','FEATURE1','FEATURE2','FEATURE3']].to_numpy(),dtype=torch.float)
        data['user', 'action', 'activity'].edge_ts = torch.tensor(graph_data['TIMESTAMP'].tolist(),dtype=torch.long)

        return data

    def create_id_dict_catagorical(self,column):
        # Creates a dictionary that creates a dictionary that using old id and creates a new id
        # Starts the ID at zero
        Old_ID = column.unique()
        New_ID = range(0,len(Old_ID))
        return dict(zip(Old_ID, New_ID))
    
    def create_id_vector_catagorical(self,column):
        # Applys the dictionary on all the Old ID to create the new ID
        oldID_To_NewID = self.create_id_dict_catagorical(column)
        return list(map(lambda oldID : oldID_To_NewID[oldID], column))

    def Credit_Card_HeteroGraph(self, root, days = None, percentage = None, truncate_size = None):
        assert (days is None) or (percentage is None), "Please provide either days or percentage, but not both."

        df = pd.read_csv(os.path.join(root,"fraudTrain.csv"))
        df = df.sort_values(by = "unix_time") #sorting by edge time

        if days is None and percentage is None:
                percentage = 1

        if days is not None:
            Min_Date = datetime.fromtimestamp(min(df["unix_time"]))
            Cut_OffDate = Min_Date + timedelta(days = days)
            Cut_OffTime = datetime.timestamp(Cut_OffDate)
            df = df.loc[df["unix_time"] <= Cut_OffTime,:]
            
        elif percentage is not None:
            df = df.iloc[:round(df.shape[0] * percentage),:]
        if truncate_size is not None:
            df = df.iloc[:truncate_size]
        column_list =["cc_num", # Customer ID (nodeID)
                    "merchant", # Merchant name (merchant ID)
                    "category", # Transaction category (Edge Feature)
                    "amt", # Amount of transaction (Edge Feature)
                    "gender", # Gender (Node feature)
                    "dob", # Date of birth (Node Feature)
                    "trans_num", # Edge ID
                    "unix_time",# Edge timestamp
                    "is_fraud",# Edge Label
                    "merch_lat", # Merchant Feature
                    "merch_long"] # Merchant Feature
                        
        #   Extract necessary columns only
        df_lc = df[column_list].copy(deep=True) 
        #   Customer ID
        df_lc["cc_num"] = self.create_id_vector_catagorical(df_lc["cc_num"])
        #   Merchant ID
        df_lc["merchant"] = self.create_id_vector_catagorical(df_lc["merchant"])
        #   Gender # Male is 1 Female is Zero
        df_lc["gender"] = (df_lc["gender"] == "M")* 1 # times one to convert bool to int
        #   Turning Date of Birth into Unix Time
        #   Could try to make an age variable but then the node would have time changing features
        df_lc["dob"] = list(map(lambda x: int(date.fromisoformat(x).strftime("%s")), df_lc["dob"]))
        #   Edge ID
        df_lc["trans_num"] = self.create_id_vector_catagorical(df_lc["trans_num"])

        #   Creating Graph
        graph = HeteroData()
        #   Customer Nodes
        Customerfeatures = ["cc_num","gender","dob"] #cc_num is needed to remove duplicates properly
        graph["customer"].x = torch.tensor(df_lc[Customerfeatures].drop_duplicates().iloc[:,1:].to_numpy(), dtype=torch.float) #Iloc to not include cc_num as feature
        #   Merchant Nodes
        Merchantfeatures = ["merchant","merch_lat","merch_long"]
        first_merchant = df_lc["merchant"].drop_duplicates().index # using the first lat and long value
        graph["merchant"].x = torch.tensor(df_lc[Merchantfeatures].iloc[first_merchant,1:].to_numpy(), dtype=torch.float)
        #   Edge 
        EdgeIndex = ["cc_num", "merchant"]
        Edgefeatures = ["category", "amt"]
        EdgeTime = ["unix_time"]
        EdgeLabel = ["is_fraud"]

        #   Convert Merhcant Category into Numeric
        df_edgefeatures = df_lc[Edgefeatures]
        df_edgefeatures['category'] = pd.factorize(df_edgefeatures['category'])[0]

        graph["customer", "transaction", "merchant"].edge_attr = torch.tensor(df_edgefeatures.to_numpy(), dtype = torch.float)
        graph["customer", "transaction", "merchant"].edge_label = torch.tensor(df_lc[EdgeLabel].to_numpy(), dtype = torch.long).squeeze()
        graph["customer", "transaction", "merchant"].edge_ts = torch.tensor(df_lc[EdgeTime].to_numpy(), dtype = torch.float).squeeze()
        graph["customer", "transaction", "merchant"].edge_index = torch.tensor(df_lc[EdgeIndex].T.to_numpy(), dtype = torch.long)

        return graph
