import torch.nn as nn
from read import Reader
import torch.autograd

class Encoder(nn.Module):
    def __init__(self, hyperParams):
        super(Encoder, self).__init__()
        self.hyperParams = hyperParams
        if hyperParams.charEmbFile == "":
            self.charEmb = nn.Embedding(hyperParams.charNUM, hyperParams.charEmbSize)
            self.charDim = hyperParams.charEmbSize
        else:
            reader = Reader()
            self.charEmb, self.charDim = reader.load_pretrain(hyperParams.charEmbFile, hyperParams.charAlpha, hyperParams.unk)
        self.charEmb.weight.requires_grad = hyperParams.charFineTune
        self.dropOut = nn.Dropout(hyperParams.dropProb)
        self.bilstm = nn.LSTM(input_size=self.charDim,
                              hidden_size=hyperParams.rnnHiddenSize,
                              batch_first=True,
                              bidirectional = True,
                              num_layers = 1,
                              dropout=hyperParams.dropProb)



    def init_hidden(self, batch = 1):
        return (torch.autograd.Variable(torch.zeros(2, batch, self.hyperParams.rnnHiddenSize)),
                torch.autograd.Variable(torch.zeros(2, batch, self.hyperParams.rnnHiddenSize)))

    def forward(self, charIndexes, hidden):
        charRepresents = self.charEmb(charIndexes)
        charRepresents = self.dropOut(charRepresents)
        output, hidden = self.bilstm(charRepresents, hidden)
        return output, hidden

