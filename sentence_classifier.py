# This is a 
import torch as pt
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

import import_ipynb
import transcript_creator


class ComplexModel(nn.Module):
    def __init__(self, input_size=3*768, hidden_size=384):
        super(ComplexModel, self).__init__()
        
        # Define layers
        self.fc1 = nn.Linear(input_size, hidden_size)  # Fully connected layer 1
        self.fc2 = nn.Linear(hidden_size, hidden_size) # Fully connected layer 2
        self.fc3 = nn.Linear(hidden_size, hidden_size) # Fully connected layer 3
        self.fc4 = nn.Linear(hidden_size, hidden_size) # Fully connected layer 4
        self.fc5 = nn.Linear(hidden_size, 1)           # Output layer
        
        self.relu = nn.ReLU()     # ReLU activation function
        self.sigmoid = nn.Sigmoid() # Sigmoid activation function
        
        self.dropout = nn.Dropout(p=0.4) # Dropout layer to prevent overfitting

    def forward(self, x):
        # Forward pass
        x = self.fc1(x)
        x = self.relu(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        
        x = self.fc3(x)
        x = self.relu(x)
        
        x = self.fc4(x)
        x = self.relu(x)
        
        x = self.dropout(x)
        
        x = self.fc5(x)
        x = self.sigmoid(x)
        
        return x
    
def triples_to_embeddings(text_triple, sent_transf):
    raw_embeddings = sent_transf.encode(text_triple)
    holder = pt.tensor(np.concatenate((raw_embeddings[0], raw_embeddings[1], raw_embeddings[2])))
    print(holder)
    return holder


        
    
def main():
    device = pt.device("mps") if pt.backends.mps.is_available() else pt.device("cpu")
    embedding_classifier = pt.load("is_conversation_embedding_classifier.pt")
    embedding_classifier.to(device)
    embedding_classifier.eval()
    # Step 2: Create the embeddings
    sent_transf = SentenceTransformer("avsolatorio/GIST-Embedding-v0", revision=None)
    sent_transf.to(device)

    transcript = transcript_creator.return_triples()

    for t in transcript.text_triples:
        triples_to_embeddings(t)


