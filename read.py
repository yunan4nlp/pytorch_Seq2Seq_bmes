from instance import  Instance
import torch
import re
import torch.nn as nn
from common import get_words
import unicodedata

class Reader:

    def normalize(self, string):
        string = re.sub(r"[0-9]", "0", string)
        return string.strip().lower()

    def readInstances(self, path, maxInst = -1):
        insts = []
        r = open(path, encoding='utf8')
        info = []

        inst = Instance()
        for line in r.readlines():
            line = line.strip()
            if line == "" and len(inst.m_char) != 0:
                if (maxInst == -1) or (maxInst > len(insts)):
                    inst.m_word = get_words(inst.m_char, inst.m_label)
                    insts.append(inst)
                else:
                    return insts
                inst = Instance()
            else:
                info = line.split(" ")
                if len(info) != 3:
                    print("error format")
                uni_char = unicodedata.normalize('NFKC', info[0])
                inst.m_char.append(uni_char)
                bi_char = unicodedata.normalize('NFKC', info[1][4:])
                inst.m_bichar.append(bi_char)
                inst.m_label.append(info[2])
        r.close()
        if len(inst.m_char) != 0:
            insts.append(inst)

        return insts

    def load_pretrain(self, file, alpha, unk):
        f = open(file, encoding='utf-8')
        allLines = f.readlines()
        indexs = {}
        info = allLines[0].strip().split(' ')
        embDim = len(info) - 1
        emb = nn.Embedding(alpha.m_size, embDim)
        oov_emb = torch.zeros(1, embDim).type(torch.FloatTensor)
        for line in allLines:
            info = line.strip().split(' ')
            wordID = alpha.from_string(info[0])
            if wordID >= 0:
                indexs[wordID] = ""
                for idx in range(embDim):
                    val = float(info[idx + 1])
                    emb.weight.data[wordID][idx] = val
                    oov_emb[0][idx] += val
        f.close()
        count = len(indexs)
        for idx in range(embDim):
            oov_emb[0][idx] /= count
        unkID = alpha.from_string(unk)
        print('UNK ID: ', unkID)
        if unkID != -1:
            for idx in range(embDim):
                emb.weight.data[unkID][idx] = oov_emb[0][idx]
        print("Load Embedding file: ", file, ", size: ", embDim)
        oov = 0
        for idx in range(alpha.m_size):
            if idx not in indexs:
                oov += 1
        print("OOV Num: ", oov, "Total Num: ", alpha.m_size,
              "OOV Ratio: ", oov / alpha.m_size)
        print("OOV ", unk, "use avg value initialize")
        return emb, embDim
