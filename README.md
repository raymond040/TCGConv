# Edge-Representation-Learning-in-Temporal-Graph

### Abstract
Deep Neural Network (DNN) has been very successful in tackling many real-world problems. However, for complex world problems where the relationship between different samples plays an essential role, vanilla deep neural networks will not be able to model and explain these problems well. Thus, the development of the Graph Neural Network (GNN) framework is crucial in combatting the limitation of vanilla DNN. GNN has achieved much success in generalising deep learning methods to complex relational data such as social networks and transaction networks. However, current GNNs mainly focus on node embeddings. Edges usually only play a supplementary role as a message-passing bridge between two adjacent nodes. Furthermore, existing edge embedding methods only apply to static and homogeneous graphs. To address these issues, this thesis introduces a new framework for GNNs, which proposes a method of learning from edges to perform edge downstream tasks, such as edge classification. The proposed framework involves converting the original graph into a conjugate graph, followed by temporal refinement to block connections that transfer information from the future to the past. Finally, temporal aggregation is performed, where information from different nodes (which are actually edges in the original graph) is aggregated using the Long Short-Term Memory (LSTM) module after sorting the nodes by time index. This allows the model to understand long-term temporal dependencies. Nevertheless, there are challenges posed in implementing the proposed framework, namely the class imbalance issue and Random Access Memory (RAM) issue. Therefore, a novel experiment method, Rolling Time Horizon, is proposed to assist in the verification of the effectiveness of the framework.  The results of this framework are remarkable (AP = 0.8084; F1 = 0.5148), surpassing existing state-of-the-art models for temporal graph neural networks like TGN and JODIE (AP = 0.6624 and 0.6045; F1 = 0.1026 and 0.1050). Overall, this thesis introduces a new framework for GNNs that is capable of learning from edges and performing edge downstream tasks with remarkable results. It has the potential to open up new avenues of research in the field of GNNs and provide a powerful tool for solving edge-related problems.


### Dependencies
A Dockerfile is provided in this repo to build your enviroment to ensure dependencies are correctly set up. Please examine the Dockerfile to get a closer look at the various version dependencies.

### Instructions 
Running the main.py script with different arguments in the command line will dictate which model is run. The code outputs performance of the model 
over epochs and over groups. The code will also print the results and the performance into a .csv file. 

#### Essential Commands

If you are running the models with HPC, then you should set HPCFlag = True at all of the models in the Models folder. Else you should set HPC_Flag = False.
The sample shell files are available at the Shell folder.
Please edit the folder and directory to your own configuration.

These commands will reproduce the model results by training a model with the same hyperparameters stated in the paper.

Running CONCAT Model
```
python3 main.py -mode train -model_type CONCAT -alpha 1.018e-02 -gamma 2 -lr 1.488e-04 -hidden_chnl 256 -num_layers 3
```
Running the All to Nodes Model
```
python3 main.py -mode train -model_type ATN -alpha 7.895e-03 -gamma 3 -lr 3.722e-05 -hidden_chnl 32 -num_layers 3
```

Running the TCGConv Model
```
python3 main.py -mode train -model_type TCGConv -alpha 4.236e-02 -gamma 2 -lr 1.000e-04 -hidden_chnl 32 -num_layers 3
```

Running the TCGConv_sum Model
```
python3 main.py -mode train -model_type TCGConv_sum -alpha 1.935e-02 -gamma 3 -lr 4.189e-05 -hidden_chnl 128 -num_layers 2
```

Running the CGConv Model
```
python3 main.py -mode train -model_type CGConv -alpha 3.439e-03 -gamma 2 -lr 4.074e-05 -hidden_chnl 128 -num_layers 3
```

Running the CGConv_sum Model
```
python3 main.py -mode train -model_type CGConv_sum -alpha 1.213e-02 -gamma 2 -lr 9.306e-05 -hidden_chnl 128 -num_layers 2
```

For running TGN and Jodie, please refer to TGN's repository https://github.com/twitter-research/tgn

#### Extra Info
To look at all the arguments in main.py please use the `h` flag
```
python main.py -h
```

