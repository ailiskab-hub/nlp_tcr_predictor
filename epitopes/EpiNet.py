import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.functional as F



class EpiTEINet(nn.Module):
    '''
    The TEINet consists of two pretrained TCRpeg encoder each for TCRs and epitopes. 
    '''
    def __init__(self, en_epi,cat_size,dropout=0,device='cuda:0',normalize=True,weight_decay=0.0):
        super().__init__()
        '''
        @en_epi: pretrained TCRpeg for epitope
        @dropout: dropout rate
        @device: the GPU device
        @normalize: whether to use layer norm
        @weight_decay: L2 regularization; 
        '''
        self.en_epi = en_epi              
        self.en_epi_model = en_epi.model
        self.device = device    
        self.projection = nn.Sequential(
            nn.Linear(cat_size, cat_size//2),
            nn.Dropout(dropout),
            nn.SELU(),            
            nn.Linear(cat_size // 2, cat_size // 4),
            nn.Dropout(dropout),
            nn.SELU(),
            nn.Linear(cat_size // 4, cat_size // 16),
            nn.Dropout(dropout),
            nn.SELU(),            
            nn.Linear(cat_size // 16, 1)
        )
        self.dropout = nn.Dropout(dropout)  
        self.normalize = normalize           
        self.weight_decay = weight_decay
        self.layer_norm_epi = nn.LayerNorm(cat_size // 2)

    def forward(self,epis):
        epi_emb = self.get_emb(epis,self.en_epi,self.en_epi_model) #b x emb
        regularization = self.weight_decay * (epi_emb.norm(dim=1).pow(2).sum())
        if self.normalize:
            epi_emb = self.layer_norm_epi(epi_emb)
        # cat = torch.cat((tcr_emb,epi_emb),dim=-1)
        if self.weight_decay == 0.0:                       
            return self.projection(epi_emb)
        else :
            return self.projection(epi_emb), regularization

    def get_emb(self,seqs,en,en_model):
        '''
        Get the latent embedding from pretrained TCRpeg
        '''               
        inputs,targets,lengths = en.aas2embs(seqs)
        inputs,targets,lengths = torch.LongTensor(inputs).to(self.device),torch.LongTensor(targets).to(self.device),torch.LongTensor(lengths).to(self.device)
        _,embedding= en_model(inputs,lengths,True)        
        embedding = embedding[:,-1,:] # B x 64
        return embedding    
    