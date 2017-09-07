from common import Alphabet

class HyperParams:
    def __init__(self):
        self.wordNUM = 0
        self.labelSize = 0

        self.unk = '-UNKNOWN-'
        self.padding = '-null-'

        self.start = '<s>'
        self.end = '-end-'

        self.charPaddingID = 0
        self.charUNKID = 0
        self.charNUM = 0

        self.bicharPaddingID = 0
        self.bicharUNKID = 0
        self.bicharNUM = 0

        self.wordPaddingID = 0
        self.wordUNKID = 0
        self.wordNUM = 0
        self.wordSTARTID = 0

        self.clip = 10
        self.maxIter = 100
        self.verboseIter = 1

        self.wordCutOff = 0
        self.wordEmbSize = 100
        self.wordFineTune = True
        self.wordEmbFile = "E:\\py_workspace\\Seq2Seq_bmes\\data\\emb_sample.txt"

        self.charCutOff = 0
        self.charEmbSize = 100
        self.charFineTune = True
        #self.charEmbFile = "E:\\py_workspace\\Seq2Seq_bmes\\data\\char.vec"
        self.charEmbFile = ""

        self.bicharCutOff = 0
        self.bicharEmbSize = 100
        self.bicharFineTune = True
        #self.bicharEmbFile = "E:\\py_workspace\\Seq2Seq_bmes\\data\\bichar.vec"
        self.bicharEmbFile = ""

        self.dropProb = 0
        self.rnnHiddenSize = 50
        self.hiddenSize = 50
        self.thread = 1
        self.learningRate = 0.001
        self.maxInstance = 5
        self.batch = 1

        self.wordAlpha = Alphabet()
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


