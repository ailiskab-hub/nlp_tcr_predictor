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

