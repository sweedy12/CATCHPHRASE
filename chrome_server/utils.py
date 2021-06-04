import torch.nn as nn
from torchcrf import CRF
import torch
from torch.utils.data import DataLoader


class LSTMcrf(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, n_layers, dropout):

        super().__init__()
        self.crf_layer = CRF(2)
        self.LSTM = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True,num_layers=n_layers)
        self.hidden2ta = nn.Linear(2*hidden_dim,2)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, seq_emb,tags,masks):
        tags = tags.permute(1,0,2)
        tags = tags.squeeze()
        tags = torch.tensor(tags,dtype=torch.long)
        masks = masks.squeeze()
        masks = torch.tensor(masks,dtype=torch.uint8)
        seq_emb = seq_emb.permute(1,0,2)
        hidden,_ = self.LSTM(seq_emb)
        # hidden_concatanated = torch.cat((h[:,0,:,:],h_n[:,1,:,:]))
        emission = self.softmax(self.hidden2ta(hidden))
        likelihood  = self.crf_layer(emission,tags,masks)
        # hidden1_full = self.dropout(packed_output1)
        return -likelihood

    def predict(self, seq_emb):
        seq_emb = seq_emb.permute(1, 0, 2)
        hidden, _ = self.LSTM(seq_emb)
        # hidden_concatanated = torch.cat((h[:,0,:,:],h_n[:,1,:,:]))
        emission = self.softmax(self.hidden2ta(hidden))
        return torch.tensor(self.crf_layer.decode(emission))


def load_pickle(path):
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)


def get_model():
    return load_pickle("lstmcrf_k1new_model")
