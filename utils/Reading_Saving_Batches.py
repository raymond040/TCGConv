#%%  Set Up
import sys
sys.path.insert(0, '/workspaces/Edge-Representation-Learning-in-Temporal-Graph/utils')
from CGT import CGT
from DatasetPrep import DatasetPrep
from TimeHorizon import Time_Batches

from zipfile import ZipFile
from ATN import allnodes
import numpy as np


import pickle
import os
from tqdm import tqdm #Progress Bar

# %%
def Save_Batches(G, num_batches, root, dataname, transform = None):
   G_Sub, G_Test = Time_Batches(G, edge_temporal=True, num_batches = num_batches, Test_Batch = True)
   for i, Sub_Graph in tqdm(enumerate(G_Sub)):
        if transform == "LG":
            GT = CGT(Sub_Graph)
        elif transform == "ATN":
            GT = allnodes(Sub_Graph)
        elif transform == None:
            GT = Sub_Graph
        else:
            raise Exception("Invalid Transformation select LG or ATN or None")
        filename = dataname + "_Sub_"  +f"{i:03}"+".obj" #Format since this will effect the ordering
        filepath = os.path.join(root,"Sub",filename)
        with open(filepath,"wb") as f:
            pickle.dump(GT, f)
    
   for i, Test_Graph in tqdm(enumerate(G_Test)):
        if transform == "LG":
            GT = CGT(Test_Graph)
        elif transform == "ATN":
            GT = allnodes(Test_Graph)
        elif transform == None:
            GT = Test_Graph
        else:
            raise Exception("Invalid Transformation select LG or ATN or None")
        filename = dataname + "_Test_"  +f"{i:03}"+".obj" #Format since this will effect the ordering
        filepath = os.path.join(root,"Test",filename)
        with open(filepath,"wb") as f:
            pickle.dump(GT, f)

def Read_Batches(root):
    """
    The Read_Batches should be used on graphs saved with the Save_Batches 
    """
    def pickle_load(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    def create_filepathlist(root, subdir):
        folder = os.path.join(root, subdir)
        files = os.listdir(folder)
        #Sort files based on number between last _ and .obj
        files.sort(reverse = False, key = lambda x: int(x.split(".")[0].split("_")[-1])) 
        filepaths = [os.path.join(folder, file) for file in files]
        return filepaths
    #Train Files
    Train_file_paths = create_filepathlist(root, "Sub")
    Train_Graphs_Map = map(pickle_load, Train_file_paths) #map is lazy won't load file until requested
    #Test Graph Map
    Test_file_paths = create_filepathlist(root, "Test")
    Test_Graphs_Map = map(pickle_load, Test_file_paths)
    
    return Train_Graphs_Map, Test_Graphs_Map

#%%
def Read_Zip(Zip_file_path):
    def pickle_load_zip(file):
        with ZipFile(Zip_file_path, 'r').open(file) as f:
            return pickle.load(f)
    with ZipFile(Zip_file_path, 'r') as zip:
        Name_list = np.array(zip.namelist())
        files = Name_list[[x.endswith(".obj") for x in Name_list]]
        sub_files = files[["Sub" in x for x in files]]
        test_files = files[["Test" in x for x in files]]
    Sub_Map = map(pickle_load_zip, sub_files)
    Test_Map = map(pickle_load_zip, test_files)
    return Sub_Map, Test_Map
#Sub_map = Read_Zip("../SubGraphs/MOOC_LineGraph_60.zip")
# #%% MOOC All to Nodes 60 Batches
# G = DatasetPrep("MOOC","../MOOC", type = "G", num_slice = 1)[0]
# root = "../SubGraphs/MOOC_AllNodes_60"
# dataname = "MOOC_AllNodesGraph"
# batches = 60
# Save_Batches(G, batches, root, dataname, transform = "ATN")

# #%% Credit Card Credit Card
# G = DatasetPrep("CC","../Credit_Card_Fraud", type = "G", num_slice = 1)[0]
# root = "../SubGraphs/CC_AllNodes_100"
# dataname = "CC_AllNodesGraph"
# batches = 100
# Save_Batches(G, batches, root, dataname, transform = "ATN")

#%%
#%% Credit Card Credit Card
# G = DatasetPrep("CC","../Credit_Card_Fraud", type = "G", num_slice = 1)[0]
# root = "../SubGraphs/CC_LineGraph_200"
# dataname = "CC_LineGraph"
# batches = 200
# Save_Batches(G, batches, root, dataname, transform = "LG")

#%% Sanity Check
# Train_G, Test_G = Read_Batches(root)
#
# for i in Train_G:
#     print(i.metadata())
# #%%
# for i in Test_G:
#     print(i.metadata())
# # %%
# Train_G, Test_G = Read_Batches(root, dataname)
# # %%
# #%% Sanity Check
# index_range_train = [(min(x["action"].original_index), max(x["action"].original_index)) for x in Train_G]
# index_range_test = [(min(x["action"].original_index), max(x["action"].original_index)) for x in Test_G]
# #%%
# G = DatasetPrep("MOOC","../MOOC", type = "G", num_slice = 1)[0]
# root = "../SubGraphs/MOOC_AllNodes_60"
# dataname = "MOOC_AllNodes"
# Save_Batches(G, 60, root, dataname, transform = "ATN")

# %%