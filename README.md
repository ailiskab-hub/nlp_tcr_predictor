# Training a language model for T-cell receptor sequences and linking them to relevant scientific literature

## Problem statement:
T-cell receptors (TCRs)  are crucial components of the adaptive immune system, responsible for recognizing and binding to specific antigens. TCR is obtained as a result of recombination of variable (V), diversity (D), and joining (J)gene segments, which combine to form a receptor. The specific binding of a TCR to an epitope of Major Histocompatibility Complex (MHC) molecules triggers a cascade of immune responses aimed at eliminating the pathogen or infected cells. The enormous variability in TCR sequences makes it difficult to find matches to TCRs of interest in existing databases such as VDJdb.  
Recent advances in machine learning offer innovative approaches such as Bidirectional Encoder Representations from Transformers (BERT) models that can be used to find and match TCR sequences through the usage of sophisticated prediction algorithms.
The primary aim of this project is to develop a model that can identify the closest TCR matches, as well as antigen and receptor binding. This can be used to associate TCRs with diseases, antigens and link them to relevant publications.

## Navigation
| Name | Description |
|-----------------|-----------------|
| data    | contains all the data that used in the project    |
| my_model.py    | file with model code, containes all required functions and methond for the next usage     |
|  EDA___model_test.ipynb   | alpha / beta subunit prediction     |
|  Metrics.ipynb   |  performance model on new data  |
|  V_J_genes.ipynb   |  V and J genes prediction   |
|  D_genes.ipynb   |  J genes prediction   |
|  Visualisation.ipynb   |  visualisation  of genes prediction   |
|  epitopes   |  fine-tuning, usage, evaluation and visualisation  |
|  ft.yml   |  required packages for running the code  |


## Results
Models perfectly coped with classification of sequences for belonging to the alpha or beta subunit of the receptor (f1 score close to 1). The best results for classifying the regions of genes V (f1-score ~ 0.6), J and D (f1-score > 0.9) whose recombination resulted in this sequence were obtained using the TCR-BERT model.

The model is able to associate a TCR with an epitope of the antigen to which the sequence will bind. This task is implemented as a classifier, relative to the most widely represented epitopes in the data. TCR BERT shows better results. Prediction by two subunits of the receptor: alpha and beta: f1 score 0.64, separately by alpha: f1 score ~ 0.44, separately by beta: f1 score ~ 0.57

The models were also trained to predict the binding of a given TCR and a given epitope. This requires first obtaining a dataset for TCR and epitope examples that do not bind to each other. And for further work, we obtained embeddings for receptor and epitope and used a fully connected neural network to classify the samples.

