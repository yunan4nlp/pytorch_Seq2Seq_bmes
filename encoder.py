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

        if hyperParams.bicharEmbFile == "":
            self.bicharEmb = nn.Embedding(hyperParams.bicharNUM, hyperParams.bicharEmbSize)
            self.bicharDim = hyperParams.bicharEmbSize
        else:
            reader = Reader()
            self.bicharEmb, self.bicharDim = reader.load_pretrain(hyperParams.bicharEmbFile, hyperParams.bicharAlpha, hyperParams.unk)
        self.bicharEmb.weight.requires_grad = hyperParams.bicharFineTune

        self.dropOut = nn.Dropout(hyperParams.dropProb)
        self.bilstm = nn.LSTM(input_size=self.charDim + self.bicharDim,
                              hidden_size=hyperParams.rnnHiddenSize,
                              batch_first=True,
                              bidirectional = True,
                              num_layers = 2,
                              dropout=hyperParams.dropProb)



    def init_hidden(self, batch = 1):
        return (torch.autograd.Variable(torch.zeros(4, batch, self.hyperParams.rnnHiddenSize)),
                torch.autograd.Variable(torch.zeros(4, batch, self.hyperParams.rnnHiddenSize)))

    def forward(self, charIndexes, bicharIndexes, hidden):
        charRepresents = self.charEmb(charIndexes)
        charRepresents = self.dropOut(charRepresents)
        bicharRepresents = self.bicharEmb(bicharIndexes)
        bicharRepresents = self.dropOut(bicharRepresents)
        concat = torch.cat((charRepresents, bicharRepresents), 2)
        output, hidden = self.bilstm(concat, hidden)
        return output, hidden

