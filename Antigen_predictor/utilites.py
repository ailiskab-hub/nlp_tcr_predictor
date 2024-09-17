from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import column_or_1d
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import re
  
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
    
    
def process_types(string, typ = 'A'):
    
    pattern = f'[A-Z]+?\d+'
    res = re.search(pattern, string)[0]
    return res#.replace(typ, '', 1)


def add_spaces(seq):
     return ' '.join(list(seq))



def generate_seqs(gene='j', chain='B', counts=1000):
    pref = '../../OLGA/olga/'
    if chain == 'B':
        params_file_name = pref + 'default_models/human_T_beta/model_params.txt'
        marginals_file_name = pref + 'default_models/human_T_beta/model_marginals.txt'
        V_anchor_pos_file = pref + 'default_models/human_T_beta/V_gene_CDR3_anchors.csv'
        J_anchor_pos_file = pref + 'default_models/human_T_beta/J_gene_CDR3_anchors.csv'
        
        genomic_data = load_model.GenomicDataVDJ()
        genomic_data.load_igor_genomic_data(params_file_name, V_anchor_pos_file, J_anchor_pos_file)

        generative_model = load_model.GenerativeModelVDJ()
        generative_model.load_and_process_igor_model(marginals_file_name)
        
        seq_gen_model = seq_gen.SequenceGenerationVDJ(generative_model, genomic_data)
        
    elif chain == 'A':
        params_file_name = pref + 'default_models/human_T_alpha/model_params.txt'
        marginals_file_name = pref + 'default_models/human_T_alpha/model_marginals.txt'
        V_anchor_pos_file = pref + 'default_models/human_T_alpha/V_gene_CDR3_anchors.csv'
        J_anchor_pos_file = pref + 'default_models/human_T_alpha/J_gene_CDR3_anchors.csv'
        
        genomic_data = load_model.GenomicDataVJ()
        genomic_data.load_igor_genomic_data(params_file_name, V_anchor_pos_file, J_anchor_pos_file)

        generative_model = load_model.GenerativeModelVJ()
        generative_model.load_and_process_igor_model(marginals_file_name)
        
        seq_gen_model = seq_gen.SequenceGenerationVJ(generative_model, genomic_data)
        
    
    
    generated = {}
    
    if gene=='j':
        for _ in range(counts):
            _, aaseq, V_in, J_in = seq_gen_model.gen_rnd_prod_CDR3()
            # res = seq_gen_model.gen_rnd_prod_CDR3()#[1::2]

            if f'TR{chain}J{str(J_in)}' not in generated.keys():
                generated[f'TR{chain}J{str(J_in)}'] = [aaseq]
            else:
                generated[f'TR{chain}J{str(J_in)}'].append(aaseq)
                
    elif gene == 'v':
        for _ in range(counts):
            _, aaseq, V_in, J_in = seq_gen_model.gen_rnd_prod_CDR3()
            
            if f'TR{chain}V{str(V_in)}' not in generated.keys():
                generated[f'TR{chain}V{str(V_in)}'] = [aaseq]
            else:
                generated[f'TR{chain}V{str(V_in)}'].append(aaseq)
                
    else:
        raise ValueError('Wrong gene type')
        
    return generated

    
def balance_minority(genes: pd.DataFrame, colu, max_count=1000):
    counts = genes[colu].value_counts()
    counts = counts.drop(counts[counts<10].index)
        
    resampled = pd.DataFrame()

    min_classes = (counts[counts<=max_count])
    i=0
    
    left_genes = pd.DataFrame()
    mean_clss = counts[counts>max_count].index#[i for i in genes[colu] if i not in min_classes]
    for cl in mean_clss:
        left_genes = pd.concat([left_genes, genes[genes[colu]==cl]])
    
    while (i < 30) and (min(min_classes) < max_count/4):
        # print('Iter '+str(i))
        i+=1
        generated_seqs_beta = generate_seqs(gene = colu, counts = max_count*100)
        generated_seqs_alpha = generate_seqs(gene = colu, chain = 'A', counts = max_count*100)
        generated_seqs = {**generated_seqs_beta, **generated_seqs_alpha}

        for min_cl in min_classes.index:
            resampled = pd.concat([resampled, genes[genes[colu] == min_cl]])
            #n = counts[min_cl] if counts[min_cl]>500 else 500
            if min_cl in generated_seqs.keys():
                for seq_tcr in generated_seqs[min_cl]:
                    resampled.loc[len(resampled.index)] = [seq_tcr, min_cl]
    
        counts_resampled = resampled[colu].value_counts()        
        min_classes = (counts_resampled[counts_resampled<=max_count])
        # print(min(min_classes))
        
    return pd.concat([left_genes, resampled])       


def mask_seqs(ori_str, maxim_len = 40):
    n = maxim_len - len(ori_str)
    substr = " [MASK]"*n
    sequences = []
    for i in range(1, 11):
        # new_seq = ori_str[:i] + substr + ori_str[i:]
        # new_seq2 = ori_str[:-i] + substr + ori_str[-i:]
        sequences.append(add_spaces(ori_str[:i]) + substr + ' '+ add_spaces(ori_str[i:]))
        # sequences.append(add_spaces(new_seq))
        
    return sequences

