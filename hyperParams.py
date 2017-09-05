class HyperParams:
    def __init__(self):
        self.wordNUM = 0
        self.bicharNUM = 0
        self.labelSize = 0

        self.unk = '-UNKNOWN-'
        self.padding = '-NULL-'

        self.start = '-start-'
        self.end = '-end-'

        self.charPaddingID = 0
        self.charUNKID = 0
        self.charNUM = 0

        self.labelSize = 0

        self.clip = 10
        self.maxIter = 100
        self.verboseIter = 1

        self.wordCutOff = 0
        self.wordEmbSize = 100
        self.wordFineTune = True
        self.wordEmbFile = ""

        self.charCutOff = 0
        self.charEmbSize = 100
        self.charFineTune = True
        self.charEmbFile = "E:\\py_workspace\\Seq2Seq_bmes\\data\\char.vec"

        self.bicharCutOff = 0
        self.bicharEmbSize = 100
        self.bicharFineTune = True
        self.bicharEmbFile = "E:\\py_workspace\\Seq2Seq_bmes\\data\\bichar.vec"

        self.dropProb = 0.5
        self.rnnHiddenSize = 50
        self.hiddenSize = 50
        self.thread = 1
        self.learningRate = 0.001
        self.maxInstance = 10
        self.batch = 3

        self.charAlpha = Alphabet()
        self.bicharAlpha = Alphabet()
        self.labelAlpha = Alphabet()
    def show(self):
        print('wordCutOff = ', self.wordCutOff)
        print('wordEmbSize = ', self.wordEmbSize)
        print('wordFineTune = ', self.wordFineTune)
        print('rnnHiddenSize = ', self.rnnHiddenSize)
        print('learningRate = ', self.learningRate)
        print('batch = ', self.batch)

        print('maxInstance = ', self.maxInstance)
        print('maxIter =', self.maxIter)
        print('thread = ', self.thread)
        print('verboseIter = ', self.verboseIter)


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