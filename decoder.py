import torch.nn as nn
import torch.autograd
from read import Reader

class Decoder(nn.Module):
    def __init__(self, hyperParams):
        super(Decoder, self).__init__()
        self.hyperParams = hyperParams
        self.linearLayer = nn.Linear(hyperParams.rnnHiddenSize * 2, hyperParams.labelSize)
        self.softmax = nn.LogSoftmax()


    def forward(self, encoder_output):
        linear = self.linearLayer(torch.cat(encoder_output, 0))
        output = self.softmax(linear)
        return output

