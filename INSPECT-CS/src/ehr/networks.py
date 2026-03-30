# This file contains the implementation of the networks used in the model
import torch.nn as nn

class PrognosisMLP(nn.Module):
    def __init__(self, input_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),        # For binary classification
        )

    def forward(self, x, return_features=False):
        features = self.net[:-1](x)
        logits = self.net[-1](features)
        if return_features:
            return logits, 0.0, {"ehr": features.cpu().detach().numpy()}
        return logits

class SupervisedAutoencoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, n_classes=1):
        super(SupervisedAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid()  # o ReLU se input non normalizzato
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes)  # 1 se binario, softmax se multi
        )
    
    def forward(self, x, return_features=False):
        embedding = self.encoder(x)
        reconstruction = self.decoder(embedding)
        logits = self.classifier(embedding)
        if return_features:
            return reconstruction, logits, {"ehr": embedding.cpu().detach().numpy()}
        return reconstruction, logits, embedding
    
def init_model(cfg):
    # instantiate the model
    if cfg.model.name == "mlp":
        model = PrognosisMLP()
    elif cfg.model.name == "sae":   
        model = SupervisedAutoencoder(
            input_dim=cfg.model.input_dim,
            embedding_dim=cfg.model.embedding_dim
        )
    else:
        raise ValueError("Invalid model name")
    return model