from read import Reader
from hyperParams import  HyperParams
from optparse import OptionParser
from instance import  Example
from encoder import  Encoder
from decoder import  Decoder
from eval import Eval
import torch.nn
import torch.autograd
import torch.nn.functional
import random

class Trainer:
    def __init__(self):
        self.char_state = {}
        self.bichar_state = {}

        self.hyperParams = HyperParams()

    def createAlphabet(self, trainInsts, devInsts, testInsts):
        print("create alpha.................")
        for inst in trainInsts:
            for c in inst.m_char:
                if c not in self.char_state:
                    self.char_state[c] = 1
                else:
                    self.char_state[c] += 1

            for bi in inst.m_bichar:
                if bi not in self.bichar_state:
                    self.bichar_state[bi] = 1
                else:
                    self.bichar_state[bi] += 1

            for l in inst.m_label:
                self.hyperParams.labelAlpha.from_string(l)


        self.addTestAlpha(devInsts)
        self.addTestAlpha(testInsts)


        self.char_state[self.hyperParams.unk] = self.hyperParams.charCutOff + 1
        self.char_state[self.hyperParams.padding] = self.hyperParams.charCutOff + 1

        self.bichar_state[self.hyperParams.unk] = self.hyperParams.bicharCutOff + 1
        self.bichar_state[self.hyperParams.padding] = self.hyperParams.bicharCutOff + 1

        self.hyperParams.charAlpha.initial(self.char_state, self.hyperParams.charCutOff)
        self.hyperParams.bicharAlpha.initial(self.bichar_state, self.hyperParams.bicharCutOff)

        self.hyperParams.charNUM = self.hyperParams.charAlpha.m_size
        self.hyperParams.charUNKID = self.hyperParams.charAlpha.from_string(self.hyperParams.unk)
        self.hyperParams.charPaddingID = self.hyperParams.charAlpha.from_string(self.hyperParams.padding)

        self.hyperParams.bicharNUM = self.hyperParams.bicharAlpha.m_size
        self.hyperParams.bicharUNKID = self.hyperParams.bicharAlpha.from_string(self.hyperParams.unk)
        self.hyperParams.bicharPaddingID = self.hyperParams.bicharAlpha.from_string(self.hyperParams.padding)
        self.hyperParams.labelSize = self.hyperParams.labelAlpha.m_size

        self.hyperParams.charAlpha.set_fixed_flag(True)
        self.hyperParams.bicharAlpha.set_fixed_flag(True)
        self.hyperParams.labelAlpha.set_fixed_flag(True)

        print('=========================================')
        print("char num: ", self.hyperParams.charNUM)
        print("char UNK ID: ", self.hyperParams.charUNKID)
        print("char Padding ID: ", self.hyperParams.charPaddingID)

        print("bichar num: ", self.hyperParams.bicharNUM)
        print("bichar UNK ID: ", self.hyperParams.bicharUNKID)
        print("bichar Padding ID: ", self.hyperParams.bicharPaddingID)

        print("label size: ", self.hyperParams.labelSize)
        print('=========================================')


    def addTestAlpha(self, testInsts):
        for inst in testInsts:
            if self.hyperParams.charFineTune == False:
                for c in inst.m_char:
                    if c not in self.char_state:
                        self.char_state[c] = 1
                    else:
                        self.char_state[c] += 1


    def instance2Example(self, insts):
        exams = []
        for inst in insts:
            example = Example()

            example.size = len(inst.m_char)
            for idx in range(example.size):
                c = inst.m_char[idx]
                charID = self.hyperParams.charAlpha.from_string(c)
                if charID == -1:
                    charID = self.hyperParams.charUNKID
                example.charIndexes.append(charID)

            for idx in range(example.size):
                l = inst.m_label[idx]
                labelID = self.hyperParams.labelAlpha.from_string(l)
                example.labelIndexes.append(labelID)
            exams.append(example)
        return exams

    def getBatchFeatLabel(self, exams):
        maxSentSize = 0
        batch = len(exams)
        for e in exams:
            if maxSentSize < len(e.labelIndexes):
                maxSentSize = len(e.labelIndexes)
        batchCharFeats = torch.autograd.Variable(torch.LongTensor(batch, maxSentSize))
        batchLabel = torch.autograd.Variable(torch.LongTensor(batch * maxSentSize))

        for idx in range(batch):
            e = exams[idx]
            for idy in range(maxSentSize):
                if idy < e.size:
                    batchCharFeats.data[idx][idy] = e.charIndexes[idy]
                else:
                    batchCharFeats.data[idx][idy] = self.hyperParams.charPaddingID

                if idy < e.size:
                    batchLabel.data[idx * maxSentSize + idy] = e.labelIndexes[idy]
                else:
                    batchLabel.data[idx * maxSentSize + idy] = 0
        return batchCharFeats, batchLabel, batch


    def train(self, train_file, dev_file, test_file, model_file):
        self.hyperParams.show()
        torch.set_num_threads(self.hyperParams.thread)
        reader = Reader()

        trainInsts = reader.readInstances(train_file, self.hyperParams.maxInstance)
        devInsts = reader.readInstances(dev_file, self.hyperParams.maxInstance)
        testInsts = reader.readInstances(test_file, self.hyperParams.maxInstance)

        print("Training Instance: ", len(trainInsts))
        print("Dev Instance: ", len(devInsts))
        print("Test Instance: ", len(testInsts))

        self.createAlphabet(trainInsts, devInsts, testInsts)

        trainExamples = self.instance2Example(trainInsts)
        devExamples = self.instance2Example(devInsts)
        testExamples = self.instance2Example(testInsts)

        self.encoder = Encoder(self.hyperParams)
        self.decoder = Decoder(self.hyperParams)

        indexes = []
        for idx in range(len(trainExamples)):
            indexes.append(idx)

        encoder_parameters = filter(lambda p: p.requires_grad, self.encoder.parameters())
        encoder_optimizer = torch.optim.Adam(encoder_parameters, lr = self.hyperParams.learningRate)

        decoder_parameters = filter(lambda p: p.requires_grad, self.decoder.parameters())
        decoder_optimizer = torch.optim.Adam(decoder_parameters, lr = self.hyperParams.learningRate)
        train_num = len(trainExamples)
        batchBlock = train_num // self.hyperParams.batch
        if train_num % self.hyperParams.batch != 0:
            batchBlock += 1
        for iter in range(self.hyperParams.maxIter):
            print('###Iteration' + str(iter) + "###")
            random.shuffle(indexes)
            self.encoder.train()
            self.decoder.train()
            train_eval = Eval()
            for updateIter in range(batchBlock):
                exams = []
                start_pos = updateIter * self.hyperParams.batch
                end_pos = (updateIter + 1) * self.hyperParams.batch
                if end_pos > train_num:
                    end_pos = train_num
                for idx in range(start_pos, end_pos):
                    exams.append(trainExamples[indexes[idx]])
                batchCharFeats, batchLabel, batch = self.getBatchFeatLabel(exams)
                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()

                encoderHidden = self.encoder.init_hidden(batch)
                encoderOutput, encoderHidden = self.encoder(batchCharFeats, encoderHidden)
                loss = 0
                decoderOutput = self.decoder(encoderOutput)

                for exam in exams:
                    for idx in range(exam.size):
                        labelID = self.getMaxIndex(decoderOutput[idx])
                        if labelID == exam.labelIndexes[idx]:
                            train_eval.correct_num += 1
                        train_eval.gold_num += 1

                loss += torch.nn.functional.nll_loss(decoderOutput, batchLabel)
                loss.backward()
                if (updateIter + 1) % self.hyperParams.verboseIter == 0:
                    print('Current: ', updateIter + 1, ", Cost:", loss.data[0], ", ACC:", train_eval.acc())
                encoder_optimizer.step()
                decoder_optimizer.step()

            self.encoder.eval()
            self.decoder.eval()

            dev_eval = Eval()
            for idx in range(len(devExamples)):
                exam = devExamples[idx]
                predict_labels = self.predict(exam)
                devInsts[idx].evalPRF(predict_labels, dev_eval)
            p, r, f = dev_eval.getFscore()
            print("precision: ", p, ", recall: ", r, ", fscore: ", f)

            test_eval = Eval()
            for idx in range(len(testExamples)):
                exam = testExamples[idx]
                predict_labels = self.predict(exam)
                testInsts[idx].evalPRF(predict_labels, test_eval)
            p, r, f = test_eval.getFscore()
            print("precision: ", p, ", recall: ", r, ", fscore: ", f)
        '''
        if iter + 1 % 10 == 0:
            self.encoder.eval()
            self.decoder.eval()
            print("Save model .....")
            self.saveModel(model_file+str(iter))
            print("Model model ok")
        '''

    def saveModel(self, model_file):
        torch.save([self.encoder, self.decoder], model_file)
        self.hyperParams.charAlpha.write(model_file + ".post")
        self.hyperParams.labelAlpha.write(model_file + ".response")

    def loadModel(self, model_file):
        self.encoder, self.decoder = torch.load(model_file)
        self.hyperParams.charAlpha.read(model_file + ".post")
        self.hyperParams.labelAlpha.read(model_file + ".response")

        self.hyperParams.postStartID = self.hyperParams.charAlpha.from_string(self.hyperParams.start)
        self.hyperParams.postEndID = self.hyperParams.charAlpha.from_string(self.hyperParams.end)
        self.hyperParams.postUnkWordID = self.hyperParams.charAlpha.from_string(self.hyperParams.unk)
        self.hyperParams.postPaddingID = self.hyperParams.charAlpha.from_string(self.hyperParams.padding)
        self.hyperParams.postWordNum = self.hyperParams.charAlpha.m_size

        self.hyperParams.responseUnkWordID = self.hyperParams.labelAlpha.from_string(self.hyperParams.unk)
        self.hyperParams.responsePaddingID = self.hyperParams.labelAlpha.from_string(self.hyperParams.padding)
        self.hyperParams.labelNum = self.hyperParams.labelAlpha.m_size


    def test(self, test_file, model_file):
        self.loadModel(model_file)
        reader = Reader()
        testInsts = reader.readInstances(test_file, self.hyperParams.maxInstance)
        testExamples = self.instance2Example(testInsts)
        for idx in range(len(testExamples)):
            self.predict(testExamples[idx])

    def predict(self, exam):
        exams = []
        exams.append(exam)
        batchCharFeats, batchLabel, batch = self.getBatchFeatLabel(exams)
        encoderHidden = self.encoder.init_hidden(batch)
        encoderOutput, encoderHidden = self.encoder(batchCharFeats, encoderHidden)
        decoderOutput = self.decoder(encoderOutput)
        sent = []
        for idx in range(exam.size):
            labelID = self.getMaxIndex(decoderOutput[idx])
            label = self.hyperParams.labelAlpha.from_id(labelID)
            sent.append(label)
        return sent



    def getMaxIndex(self, decoder_output):
        max = decoder_output.data[0]
        maxIndex = 0
        for idx in range(1, self.hyperParams.labelSize):
            if decoder_output.data[idx] > max:
                max = decoder_output.data[idx]
                maxIndex = idx
        return maxIndex





parser = OptionParser()
parser.add_option("--train", dest="trainFile",
                  help="train dataset")

parser.add_option("--dev", dest="devFile",
                  help="dev dataset")

parser.add_option("--test", dest="testFile",
                  help="test dataset")

parser.add_option("--model", dest="modelFile",
                  help="model file")
parser.add_option(
    "-l", "--learn", dest="learn", help="learn or test", action="store_false", default=True)

random.seed(0)
torch.manual_seed(0)
(options, args) = parser.parse_args()
l = Trainer()
if options.learn:
    l.train(options.trainFile, options.devFile, options.testFile, options.modelFile)
else:
    l.test(options.testFile,options.modelFile)


