# Edge-Representation-Learning-in-Temporal-Graph

### Abstract
Graph Neural Networks (GNNs) have been very successful in generalizing deep learning methods to relational data such as social networks. However, current GNNs primarily focus on learning node representations while edges are only used as auxiliary information. Despite the importance of edge representation learning and edge-level downstream tasks such as edge classification, little attention has been paid to them. Furthermore, existing edge embedding methods predominantly apply to static graphs. To bridge these gaps, we present a novel framework, Temporal Conjugate Graph Convolution (TCGConv), to learn edge representations in temporal graphs. TCGConv consists of Conjugate Graph Transformation (CGT), which reverses the role of nodes and edges, and Long Short-Term Memory (LSTM) aggregator, which captures temporal information during the aggregation process. We then conduct extensive experiments on a public transaction dataset to validate our framework. These experiments show that our proposed model significantly outperforms the baseline models, which are impractical in real-time applications as they disregard the temporal constraints by peeking into the future, and its variants, demonstrating the effectiveness of our model to incorporate temporal information.



### Dependencies
A Dockerfile is provided in this repo to build your enviroment to ensure dependencies are correctly set up. Please examine the Dockerfile to get a closer look at the various version dependencies.

### Instructions 
Running the main.py script with different arguments in the command line will dictate which model is run. The code outputs performance of the model 
over epochs and over groups. The code will also print the results and the performance into a .csv file. 

#### Essential Commands
These commands will reproduce our model results by training a model with the same parameters we use from our paper

Running CONCAT Model
```
python main.py -mode train -model_type CONCAT -alpha 1.018e-02 -gamma 2 -lr 1.488e-04 -hidden_chnl 256 -num_layers 3
```
Running the All to Nodes Model
```
python main.py -mode train -model_type ATN -alpha 7.895e-03 -gamma 3 -lr 3.722e-05 -hidden_chnl 32 -num_layers 3
```

Running the TCGConv Model
```
python main.py -mode train -model_type TCGConv -alpha 4.236e-02 -gamma 2 -lr 1.000e-04 -hidden_chnl 32 -num_layers 3
```

Running the TCGConv_sum Model
```
python main.py -mode train -model_type TCGConv_sum -alpha 1.935e-02 -gamma 3 -lr 4.189e-05 -hidden_chnl 128 -num_layers 2
```

Running the CGConv Model
```
python main.py -mode train -model_type CGConv -alpha 3.439e-03 -gamma 2 -lr 4.074e-05 -hidden_chnl 128 -num_layers 3
```

Running the CGConv_sum Model
```
python main.py -mode train -model_type CGConv_sum -alpha 1.213e-02 -gamma 2 -lr 9.306e-05 -hidden_chnl 128 -num_layers 2
```

#### Extra Info
To look at all the arguments in main.py please use the `h` flag
```
python main.py -h
```




**Use of this source code is governed by an MIT-style license that can be found in the LICENSE file**
