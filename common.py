def getMaxIndex(hyperParams, decoder_output):
    max = decoder_output.data[0]
    maxIndex = 0
    for idx in range(1, hyperParams.labelSize):
        if decoder_output.data[idx] > max:
            max = decoder_output.data[idx]
            maxIndex = idx
    return maxIndex

def is_end_label(label):
    end = ['s', 'S', 'e', 'E']
    if(len(label) < 3):
        return False
    else:
        return (label[0] in end) and label[1] == '-'

def is_start_label(label):
    start = ['b', 'B', 's', 'S']
    if(len(label) < 3):
        return False
    else:
        return (label[0] in start) and label[1] == '-'

def cleanLabel(label):
    start = ['B', 'b', 'M', 'm', 'E', 'e', 'S', 's', 'I', 'i']
    if len(label) > 2 and label[1] == '-':
        if label[0] in start:
            return label[2:]
    return label

def is_continue_label(label, startLabel, distance):
    if distance == 0:
        return True
    if len(label) < 3:
        return False
    if distance != 0 and is_start_label(label):
        return False
    if (startLabel[0] == 's' or startLabel[0] == 'S') and startLabel[1] == '-':
        return False
    if cleanLabel(label) != cleanLabel(startLabel):
        return False
    return True

def get_ent(labels):
        idx = 0
        idy = 0
        endpos = -1
        ent = []
        while(idx < len(labels)):
            if (is_start_label(labels[idx])):
                idy = idx
                endpos = -1
                while(idy < len(labels)):
                    if not is_continue_label(labels[idy], labels[idx], idy - idx):
                        endpos = idy - 1
                        break
                    endpos = idy
                    idy += 1
                ent.append(cleanLabel(labels[idx]) + '[' + str(idx) + ',' + str(endpos) + ']')
                idx = endpos
            idx += 1
        return ent

def get_words(chars, labels):
    idx = 0
    idy = 0
    endpos = -1
    words = []
    while(idx < len(labels)):
        if (is_start_label(labels[idx])):
            idy = idx
            endpos = -1
            while(idy < len(labels)):
                if not is_continue_label(labels[idy], labels[idx], idy - idx):
                    endpos = idy - 1
                    break
                endpos = idy
                idy += 1
            tmp_word = ""
            for i in range(idx, endpos + 1):
                tmp_word += chars[i]
            words.append(tmp_word)
            idx = endpos
        idx += 1
    return words

class Alphabet:
    def __init__(self):
        self.max_cap = 1e8
        self.m_size = 0
        self.m_b_fixed = False
        self.id2string = []
        self.string2id = {}

    def from_id(self, qid, defineStr = ''):
        if int(qid) < 0 or self.m_size <= qid:
            return defineStr
        else:
            return self.id2string[qid]

    def from_string(self, str):
        if str in self.string2id:
            return self.string2id[str]
        else:
            if not self.m_b_fixed:
                newid = self.m_size
                self.id2string.append(str)
                self.string2id[str] = newid
                self.m_size += 1
                if self.m_size >= self.max_cap:
                    self.m_b_fixed = True
                return newid
            else:
                return -1

    def clear(self):
        self.max_cap = 1e8
        self.m_size = 0
        self.m_b_fixed = False
        self.id2string = []
        self.string2id = {}

    def set_fixed_flag(self, bfixed):
        self.m_b_fixed = bfixed
        if (not self.m_b_fixed) and (self.m_size >= self.max_cap):
            self.m_b_fixed = True

    def initial(self, elem_state, cutoff = 0):
        for key in elem_state:
            if  elem_state[key] > cutoff:
                self.from_string(key)
        self.set_fixed_flag(True)

    def initialFromEmb(self, path):
        embFile = open(path, encoding='utf-8', mode='r')
        for line in embFile.readlines():
            info = line.split(" ")
            self.from_string(info[0])
        embFile.close()

    def write(self, path):
        outf = open(path, encoding='utf-8', mode='w')
        for idx in range(self.m_size):
            outf.write(self.id2string[idx] + " " + str(idx) + "\n")
        outf.close()

    def read(self, path):
        inf = open(path, encoding='utf-8', mode='r')
        for line in inf.readlines():
            info = line.split(" ")
            self.id2string.append(info[0])
            self.string2id[info[0]] = int(info[1])
        inf.close()
        self.set_fixed_flag(True)
        self.m_size = len(self.id2string)
