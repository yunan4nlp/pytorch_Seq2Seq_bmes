import torch.nn as nn
import torch.autograd
from read import Reader
from common import getMaxIndex
from common import is_end_label

class Decoder(nn.Module):
    def __init__(self, hyperParams):
        super(Decoder, self).__init__()

        reader = Reader()
        self.wordEmb, self.wordDim = reader.load_pretrain(hyperParams.wordEmbFile, hyperParams.wordAlpha, hyperParams.unk)

        self.wordEmb.weight.requires_grad = hyperParams.wordFineTune
        self.dropOut = torch.nn.Dropout(hyperParams.dropProb)
        self.lastWords = []
        self.hyperParams = hyperParams
        #self.linearLayer = nn.Linear(hyperParams.rnnHiddenSize * 2, hyperParams.labelSize)

        self.linearLayer = nn.Linear(hyperParams.rnnHiddenSize * 2 + self.wordDim, hyperParams.labelSize)

        self.softmax = nn.LogSoftmax()


    def forward(self, batch, encoder_output, exams):
        sent_len = encoder_output.size()[1]
        self.lastWords = []
        batch_labels = []
        last_word_indexes = torch.autograd.Variable(torch.LongTensor(batch))
        output = []
        for idy in range(batch):
            self.lastWords.append('<s>')
            batch_labels.append([])
            output.append([])
            last_word_indexes.data[idy] = self.hyperParams.wordSTARTID

        for idx in range(sent_len):
            char_presentation = encoder_output.permute(1, 0, 2)[idx]
            last_word_presentation = self.wordEmb(last_word_indexes)
            last_word_presentation = self.dropOut(last_word_presentation)
            concat = torch.cat((char_presentation, last_word_presentation), 1)

            hidden = self.linearLayer(concat)
            #print(hidden.size())
            batch_hidden = torch.chunk(hidden, batch, 0)
            #print(batch_out[0].size())

            for idy in range(batch):
                output[idy].append(batch_hidden[idy])
                labelID = getMaxIndex(self.hyperParams, hidden[idy])
                label = self.hyperParams.labelAlpha.from_id(labelID)
                batch_labels[idy].append(label)
                self.prepare(exams[idy].m_char, idx, batch_labels[idy], idy)

                wordID = self.hyperParams.wordAlpha.from_string(self.lastWords[idy])
                if wordID < 0:
                    wordID = self.hyperParams.wordUNKID
                last_word_indexes.data[idy] = wordID
        for idy in range(batch):
            output[idy] = torch.cat(output[idy], 0)
        output = torch.cat(output, 0)
        output = self.softmax(output)
        return output
        '''


        linear = self.linearLayer(torch.cat(encoder_output, 0))
        output = self.softmax(linear)
        print(output.size())
        return output
        '''

    def prepare(self, m_char, index, labels, batchIndex):
        if index < len(m_char):
            if labels[index][0] == 'S' or labels[index][0] == 's':
                self.lastWords[batchIndex] = m_char[index]
            if labels[index][0] == 'E' or labels[index][0] == 'e':
                tmp_word = m_char[index]
                idx = index - 1
                while (idx >= 0) and (labels[idx][0] == 'M' or labels[idx][0] == 'm'):
                    tmp_word += m_char[idx]
                    idx -= 1
                if idx >= 0 and (labels[idx][0] == 'B' or labels[idx][0] == 'b'):
                    tmp_word += m_char[idx]
                    self.lastWords[batchIndex] = tmp_word[::-1]
        else:
            self.lastWords[batchIndex] = '-null-'
