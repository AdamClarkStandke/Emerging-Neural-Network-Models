# Graph Neural Networks
[Graph Neural Networks(GNNs)](https://en.wikipedia.org/wiki/Graph_neural_network#cite_note-:1-43) are a type of neural network architecture that are designed for tasks whose inputs are graphs. I will be implementing various chapters of the book [Hands-On Graph Neural Network Using Python](https://github.com/PacktPublishing/Hands-On-Graph-Neural-Networks-Using-Python) written by Maxime Labonne. 

## Chapter 10: Predicting edges with Homo VGAEs
Ch 10 deals with predicting links/edges of homogeneous graphs by using Variational Graph Autoencoders(VGAE). VGAE were introduced in the paper [Variational Graph Auto-Encoders](https://arxiv.org/abs/1611.07308) and implemented using [torch_geometric](https://pytorch-geometric.readthedocs.io/en/2.7.0/generated/torch_geometric.nn.models.VGAE.html). 

VGAE is composed of an encoder and decoder. The encoder is composed of three [graph convolutional networks(GCN)](https://arxiv.org/abs/1609.02907) in which the first GCN is the base/shared layer that produces the hidden output i.e., h for the other two GCN heads. It takes in a matrix of nodes and their features i.e., X and the adjacency/connectivity matrix i.e., A to produce the hidden output. After doing so, the hidden output is used as input to the two GCN heads. One GCN head learns the mean of the latent normal distribution i.e., mu and the other GCN head learns the log std of the latent normal distribution i.e., sigma.  Also the [Reparameterization trick](https://en.wikipedia.org/wiki/Reparameterization_trick) is used during this stage to backpropagate the gradients. 

The decoder is an  Inner Product Decoder which takes in the latent normal distribution i.e., Z and computes the edge probabilities for the given node-pairs using the logistic sigmoid function to output the predicted adjacency/connectivity matrix i.e., A. Lastly, the evidence lower bound (ELBO) is calculated which is comprised of the reconstruction loss (BCE loss) and the Kullback-Leibler (KL) divergence loss. 

[VGAE](https://github.com/AdamClarkStandke/GraphNeuralNetworks/blob/main/Chapter10_prediEdgeLinks_GVAE.ipynb) is implemented on the [Cora dataset](https://graphsandnetworks.com/the-cora-dataset/) from Planetoid


## Chapter 11: Generating Graphs with Homo GNNs

## Chapter 12: Hetero Graphs and Hetro GNNs
