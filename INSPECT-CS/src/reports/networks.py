# This file contains the implementation of the networks used in the model
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

# Soft Attention Mechanism
class SoftAttentionLayer(nn.Module):
    def __init__(self, input_size):
        super(SoftAttentionLayer, self).__init__()
        self.input_size = input_size
        self.classifier = nn.Linear(input_size, 1)
        # initialize self.classifier with mean 0 and bias 1
        with torch.no_grad():
            self.classifier.weight.fill_(0)
            self.classifier.bias.fill_(1)

    def forward(self, inputs): 
        # get batch size and sequence length from inputs
        batch_size, seq_len, _ = inputs.size()
        # calculate attention scores
        scores = self.classifier(inputs).view(batch_size, seq_len)
        # apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=1)

        # compute the weighted mean
        attention_weights = attention_weights.unsqueeze(2)
        weighted_sum = torch.bmm(inputs.transpose(1, 2), attention_weights).squeeze(2)

        return weighted_sum 
    
    def get_attention_weights(self, inputs):
        # get batch size and sequence length from inputs
        batch_size, seq_len, _ = inputs.size()
        scores = self.classifier(inputs).view(batch_size, seq_len)
        attention_weights = F.softmax(scores, dim=1)
        return attention_weights

class HEANetwork(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # settings
        self.enc_size = cfg.model.embed_size

        # shared soft attention layer (token and sentence levels)
        self.soft_attention = SoftAttentionLayer(self.enc_size)

        # output layer (mlp) con BatchNorm1d
        self.mlp = nn.Sequential(
            nn.Linear(self.enc_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, inputs, return_features=False):
        """
        inputs: Embeddings of the reports 
        """
        inputs = inputs  # (batch_size, num_sentence, seq_length, hidden_dim)
        
        x = inputs.view(-1, inputs.size(2), inputs.size(3))  # (batch_size*num_sentence, seq_length, hidden_dim)
        
        x = self.soft_attention(x)  # (batch_size*num_sentence, hidden_dim)
        
        x = x.view(inputs.size(0), inputs.size(1), -1)  # (batch_size, num_sentence, hidden_dim)
        
        features = self.soft_attention(x)  # (batch_size, hidden_dim)

        logits = self.mlp(features)  # (batch_size, 1)
        
        if return_features:
            return logits, 0.0, {"report": features.cpu().detach().numpy()}
            
        return logits