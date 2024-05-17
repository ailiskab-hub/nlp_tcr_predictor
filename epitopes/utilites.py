from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import column_or_1d
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
  
from Bio.Align import substitution_matrices
from Bio import Align



def draw_heatmap(pred_labels, ans, l_e, n_cl = None, show=True, return_mtr=True):
    pred_genes = l_e.inverse_transform(pred_labels)
    ans_genes = l_e.inverse_transform(list(ans))
    
    if not n_cl:
        classes = l_e.classes_
    else:
        classes = l_e.classes_[:n_cl]
    
    matrix = pd.DataFrame(data = 0, columns= classes, index= classes)
    for i in range(len(ans_genes)):
        # print(ans_genes[i]=='RLRAEAQVK', pred_genes[i]=='RLRAEAQVK')
        matrix.loc[ans_genes[i], pred_genes[i]] += 1
    # print(matrix)
    matrix_norm = MinMaxScaler().fit_transform(matrix.T)
    matrix_norm = pd.DataFrame(data = matrix_norm.T, columns= classes, index= classes)
    # print(matrix_norm)
    if show:
        fig, ax = plt.subplots(figsize=(11,9)) 
        matrix_norm = pd.DataFrame(data = matrix_norm, columns= classes, index= classes)
        sns.heatmap(matrix_norm, cmap="Greens")
    if return_mtr:
        return matrix_norm
    

class MyLabelEncoder(LabelEncoder):
    def fit(self, y):
        y = column_or_1d(y, warn=True)
        self.classes_ = pd.Series(y).unique()
        return self
    
    
def add_spaces(seq):
     return ' '.join(list(seq))
 
    
def balance_majority(genes: pd.DataFrame, colu, min_count=0, max_count=1500):
    counts = genes[colu].value_counts()
    counts = counts.drop(counts[min_count>counts].index)
    resampled = pd.DataFrame()
    maj_clss = (counts[counts>max_count]).index
    left_genes = pd.DataFrame()
    mean_clss = counts[(counts<max_count) & (min_count<counts)].index#[i for i in genes[colu] if i not in min_classes]
    for cl in mean_clss:
        #print(cl)
        left_genes = pd.concat([left_genes, genes[genes[colu]==cl]])
    for maj_cl in maj_clss:        
        resampled = pd.concat([resampled, resample(genes[genes[colu] == maj_cl], replace=False, n_samples=max_count, random_state=42)])
    return pd.concat([left_genes, resampled])


def norm(dist_matr):
    dfmax, dfmin = np.array(dist_matr).max(), np.array(dist_matr).min()
    dist_matr_norm = (dist_matr - dfmin)/(dfmax - dfmin)
    return dist_matr_norm


def mist_dist_epit(matrox_norm, epitopes, rt_dist_mtr=False):
    dist_matr = pd.DataFrame(data = 0, columns=epitopes, index= epitopes)
    for i in epitopes:
        for j in epitopes:
            aligner = Align.PairwiseAligner()
            aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
            alignments = aligner.align(i, j)
            score = alignments[0].score
            distance = len(i) + len(j) - 2 * score
            # distance = score/(len(i)+len(j))

            dist_matr.loc[i, j] = 2*score/((len(i)*len(j)))#distance
            
    
    # print(matrox_norm)
    dist_matr_norm = norm(dist_matr)#MinMaxScaler().fit_transform(dist_matr)
    dist_matr_norm = pd.DataFrame(data = dist_matr_norm, columns= epitopes, index= epitopes)
    dist_matr_norm = dist_matr_norm+0.000001
    m1 = matrox_norm/dist_matr_norm
    # print(m1)
    m1 = m1.fillna(0)
    m1_norm = MinMaxScaler().fit_transform(m1.T)
    # print(m1_norm)
    fig, ax = plt.subplots(figsize=(11,9)) 
    m1_norm = pd.DataFrame(data = m1_norm.T, columns= epitopes, index= epitopes)
    sns.heatmap(m1_norm, cmap="Greens")
    
    if rt_dist_mtr:
        return dist_matr_norm 
    