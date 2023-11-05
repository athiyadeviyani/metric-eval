import torch
from torch import nn, optim

import torch.nn.functional as F


# Model implementation

class MatrixFactorization(nn.Module):
    def __init__(self, num_factors, num_users, num_items, sparse=False, **kwargs):
        super().__init__()
        self.sparse = sparse
        
        self.user_embedding = nn.Embedding(num_users, num_factors, sparse=sparse)
        self.user_bias = nn.Embedding(num_users, 1, sparse=sparse)
        
        self.item_embedding = nn.Embedding(num_items, num_factors, sparse=sparse)
        self.item_bias = nn.Embedding(num_items, 1, sparse=sparse) 

        for param in self.parameters():
            nn.init.normal_(param, std=0.01)   
            
    def forward(self, user_id, item_id):
        Q = self.user_embedding(user_id)
        bq = self.user_bias(user_id).flatten()

        I = self.item_embedding(item_id)
        bi = self.item_bias(item_id).flatten()

        return (Q*I).sum(-1) + bq + bi
    
