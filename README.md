# Machine Intelligence Lab

## Week 1 - Numpy and Pandas

Numpy and Pandas are one of the most important libraries in python that helps in implementation of various machine learning algorithm. 
In this week the task is to understand these libraries in depth and implement the few function using these libraries.

## Week 2 - Search Algorithms

Search Algorithms aim at navigating from a start state to a goal state by transitioning through intermediate states. It also consists of a state space which is a set of all possible states where you can be. 
There are many informed and uninformed search algorithms that exist and are very popular. 
A* search, Uniform Cost search (UCS), Depth First Search (DFS), Greedy Search to name a few. 
In this assignment you are are required to write two functions A_star_Traversal and DFS_Traversal which implements the respective algorithm

## Week 3 - Decision Tree Classifier
     
Decision Trees are one of the easiest and popular classification algorithms to understand and interpret. The goal of using a Decision Tree is to create a training model that can be used to predict the class or value of the target variable by learning simple decision rules inferred from prior data. 
The primary challenge in the decision tree implementation is to identify which attributes to consider as the root node at each level. Handling this is known as attribute selection. The ID3 algorithm builds decision trees using a top-down greedy search approach through the space of possible branches with no backtracking. It always makes the choice that seems to be the best at that moment. 
Attribute selection in the ID3 algorithm involves various steps such as computing entropy, information gain and selecting the most appropriate attribute as the root node. 
 
In this assignment you are are required to prepare a module that will help any machine learning fresher to use to calculate these heuristic on any categorical attributed data

## Week 4 - k Nearest Neighbours 

k-Nearest Neighbours is one of the most famous supervised classification and regression algorithms. The beauty of the algorithm is that we can observe it in our real life, i.e., our nature is to be close with people who are the most like us. 
 
In this assignment you are required to prepare a Python class KNN which can be used by anyone for classification. The detailed instructions are given to you in this document.

## Week 5 – Artificial Neural Networks 
 
In this week’s experiment, you will be mimicking one of the most popular Python frameworks used for deep learning, TensorFlow. TensorFlow stores all operations internally in a graph data structure called a “computation graph”. A computation graph is a Directed Acyclic Graph (DAG) in which each node represents a data value (in our case, a multidimensional array called a Tensor) and edges of the graph represent the flow of data through binary operations that is performed on 2 input Tensors and returns a single output Tensor. (Note the operations in the below diagrams are represented as nodes, this is just for your understanding. In our implementation we will not create a node for an operation). 
 
Your task in this week’s experiment is to write functions that, given a computation graph, compute the gradient of a Tensor variable (say ‘x’) with respect to the leaf nodes of the computation graph that created the Tensor ‘x’. That is to implement ‘chain rule’ using computation graphs  
For the purposes of this week’s experiment, the only operations that can be carried out on 2 Tensors are tensor addition and tensor multiplication (these are the same as matrix addition and matrix multiplication respectively).

## Week 6 – Support Vector Machines 
  
In this week’s experiment, you will implement a Support Vector Machine classifier using one of the most popular machine learning frameworks in Python, called scikit-learn. 
Scikit-learn is one of the most widely used and fully featured machine learning frameworks that, apart from offering a wide variety of machine learning models, also offers many classes for pre-processing data. 
You are expected to use the pre-processing steps of your choice along with the SVM classifier to create a pipeline (explained later) which will be used to automate the entire process of training and evaluating the model you build. 

## Week 7 - Adaboost 
 
An ensemble method is a technique that combines the predictions from multiple machine learning algorithms together .To make more accurate predictions than any individual model. One such algorithm is Adaboost. 
 
In week 7 you are required to code for the class AdaBoost which will implement Adaboost algorithm. 

## Week 8 - Hidden Markov Models 

In this week’s experiment, you will implement the Viterbi algorithm for decoding a sequence of observations to find the most probable sequence of internal states that generated the observations. 
Remember that a Hidden Markov Model (HMM) is characterized by the set of internal states, the set of observations, the transition matrix (between the internal states), the emission matrix (between the internal states and the emitted observations) and the initial distribution of the internal states

## Week 9 - k-Means Clustering  
 
kMeans Clustering is one of the popular unsupervised learning algorithm 
In week 9 you are required to code for the class KMeansClustering which will implement kMeans 
Algorithm.

## Image Classification with CNN on CIFAR10

In this week’s experiment, you will be making a convolutional neural network for image classification on the CIFAR10 dataset. The CIFAR-10 dataset consists of 60000 32x32 color images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. 
A convolutional base consists of a stack of 2Dconvolutional, maxpooling layer to name a few. 
The output tensor from the convolutional base is passed on to one or more dense layers to perform classification.

## Genetic Algorithm - XOR Gate
Build and Train a neural network using Genetic Algorithm to realise the functionality of XOR gate.
