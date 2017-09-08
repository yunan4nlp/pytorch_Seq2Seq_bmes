from common import get_ent

class Instance:
    def __init__(self):
        self.m_char = []
        self.m_word = []
        self.m_bichar = []
        self.m_label = []

    def show(self):
        print(self.m_char)
        print(self.m_label)


    def evalPRF(self, predict_labels, eval):
        gold_ent = get_ent(self.m_label)
        predict_ent = get_ent(predict_labels)
        eval.predict_num += len(predict_ent)
        eval.gold_num += len(gold_ent)

        for p in predict_ent:
            if p in gold_ent:
                eval.correct_num += 1





class Example:
    def __init__(self):
        self.m_char = []
        self.m_word = []
        self.m_label = []
        self.charIndexes = []
        self.bicharIndexes = []
        self.size = 0

        self.labelIndexes = []

    def show(self):
        print(self.charIndexes)
        print(self.bicharIndexes)
        print(self.labelIndexes)
        print(self.size)

