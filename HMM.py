from nltk import *
import numpy as np
from nltk.corpus import brown
from nltk.corpus import treebank
from nltk.corpus import conll2000
from nltk.corpus import conll2002
from nltk.corpus import alpino
from nltk.corpus import floresta
from sklearn.linear_model import LinearRegression
import copy
import argparse
import csv
from cmath import log10

''' ############ Argument handling ###########'''

parser = argparse.ArgumentParser(description='HMM tagging.')
parser.add_argument('corpus', metavar='corpus', type=str,
                    help='A string which is the name of the corpus. Which could be either Brown, Conll2000, Conll2002, Alpino, Floresta, or Treebank')

parser.add_argument('trainingSize', metavar='trainingSize', type=int,
                    help='An integer for the trainingSize. The program will start training until reaching certain amount(trainingSize) of \
                    sentences in selected corpus. For example, if the trainingSize = 500, the sentences position 0 to 499 in corpus will be used for training.')

parser.add_argument('testingSize', metavar='testingSize', type=int,
                    help='An integer for the testingSize. The amount of testing sentences in the selected corpus which will start testing at\
                     the position after the trainingSize. For example, if the trainingSize = 500, testingSize is 300, the sentences position 500 to 799 \
                     in corpus will be used for testing')

parser.add_argument('-u', action='store_true',
                    help='Use the universal tag, if it is available')

parser.add_argument('-gt', action='store_true',
                    help='Use Good-Turing smoothing technique in computing transition probability,unless the Laplace technique \
                    will be used instead.')

args = parser.parse_args()

''' ############ training Part #############'''

if args.u is True:
    tagType = 'universal'
else:
    tagType = ''
if args.corpus.lower() == 'brown':
    corpus = brown
elif args.corpus.lower() == 'conll2000':
    corpus = conll2000
elif args.corpus.lower() == 'conll2002':
    corpus = conll2002
    tagType = ''
elif args.corpus == 'alpino':
    corpus = alpino
    tagType = ''
elif args.corpus.lower() == 'floresta':
    corpus = floresta
    tagType = ''
elif args.corpus.lower() == 'treebank':
    corpus = treebank
    tagType = ''
else:
    print 'This corpus is currently unavailable'
    quit()


trainingSents = corpus.tagged_sents(tagset=tagType)
trainingSize = args.trainingSize
testingSize = args.testingSize
if trainingSize+testingSize > len(trainingSents):
    print "The amount of sentecnces of this corpus is not enough"
    print "This corpus has only %d sentences" % (len(trainingSents))
    quit()
print 'Start training \n ...... Please wait......'
words = []
tags = []
bigrams = []
freqTransition = {}
freqEmission = {}

''' Add start and end tags at the beginning and the end of each sentence'''
for trainingSent in trainingSents[:trainingSize]:
    words += ['<s>'] + [w for (w, _) in trainingSent] + ['</s>']
    tags += ['<s>'] + [ t for (_, t) in trainingSent] + ['</s>']

''' Count individual word & tag'''
countWords = FreqDist(words)
countTags = FreqDist(tags)

''' Find words occurring once'''

for index, w in enumerate(countWords.keys()):
    if countWords[w] == 1:
        words[index] = 'UNK'


print 'Replace words occur once with \"UNK\"'

wordTagPair =[]
for tag,word in zip(tags,words):
    wordTagPair += [(tag,word)]
print 'Construct emission probability table'

''' count emission'''
for pair in wordTagPair:
    tag = pair[0]
    word = pair[1]
    if tag not in freqEmission.keys():
        freqEmission[tag] = {}

    if word not in freqEmission[tag].keys():
        freqEmission[tag][word] = 1
    else:
        freqEmission[tag][word] += 1


''' Copy dictionary structure from freqEmission '''
probEmission = copy.deepcopy(freqEmission)

''' Calculate emission probability'''
for tag in probEmission.keys():
    for word in probEmission[tag].keys():
        probEmission[tag][word] = 1.0 * freqEmission[tag][word] / countTags[tag]

probTransition = {}
linreg = LinearRegression()
Nc = {}

bigrams = ngrams(tags, 2)
''' Count bigrams(transition) of tags'''
for bigram in bigrams:
    prev = bigram[0]
    current = bigram[1]
    if prev not in freqTransition.keys():
        freqTransition[prev] = {}
    if current not in freqTransition[prev].keys():
        freqTransition[prev][current] = 1
    else:
        freqTransition[prev][current] += 1


def goodTuringLinearReg():
    print "Applying Good-Turing smoothing"
    bigrams = ngrams(tags, 2)
    freqBigrams = FreqDist(bigrams)
    for key in freqBigrams:
        if freqBigrams[key] not in Nc.keys():
            Nc[freqBigrams[key]] = 1
        else:
            Nc[freqBigrams[key]] +=1

    '''####### Linear regression ####### '''
 
    x = [np.real(log10(c)) for c in Nc.keys()]
    x = np.c_[np.ones_like(x),x]
    y = [np.real(log10(Nc[key])) for key in Nc.keys()]
    #print y
    linreg.fit(x,y)

''' Given value of c return Value of Nc'''
def getNc(c):
    #print c]
    x = [log10(c)]
    x = np.c_[np.ones_like(x),x]

    y_predict = linreg.predict(x)
    return pow(10,y_predict[0])
#

'''Calculate probabilities
    probTransition[previous][current] applying Good-Turing smoothing
'''
def findTransitionProb_LP():
    print "Applying Laplace smoothing"
    for prev in countTags.keys():
        for current in countTags.keys():
            if current not in probTransition.keys():
                probTransition[current] = {}
            if (prev not in freqTransition.keys()) or current not in freqTransition[prev]:
                freq = 0
            else:
                freq = freqTransition[prev][current]
            ''' Use Laplace smoothing equation finding probability'''
            probTransition[current][prev] = (1.0 * freq + 1) / (countTags[prev] + len(tags))
      

def findTransitionProb_GTR():
    N = sum([1 for b in ngrams(tags, 2)])
    for prev in countTags.keys():
        for current in countTags.keys():
            if current not in probTransition.keys():
                probTransition[current] = {}
            if (prev not in freqTransition.keys()) or current not in freqTransition[prev]:
                freq = 0
            else:
                freq = freqTransition[prev][current]
            if freq ==0:
                probTransition[current][prev] = (getNc(1))*1.0 / (N)
            else:
                probTransition[current][prev] = ((freq+1)*(getNc(freq+1))*1.0 / (getNc(freq)))/(N)

''' Function finding Alpha '''
def getTransitionProbs(current, previous):
    if current not in probTransition.keys():
        return 0.0
    else:
        if previous not in probTransition[current].keys():
            return 0.0
        else:
            return probTransition[current][previous]


''' Function finding Beta '''
def getEmissionProbs(tag, word):
    if word not in probEmission[tag].keys():
        return 0.0
    else:
        return probEmission[tag][word]

print 'Construct transition probability table'

if args.gt is True:
    goodTuringLinearReg()
    findTransitionProb_GTR()
else:
    findTransitionProb_LP()

''' ################### Testing part ########################'''

print 'Start testing \n ...... Please wait......'

testingSents = corpus.sents()[trainingSize:trainingSize + testingSize]
testingTaggedSents = corpus.tagged_sents(tagset=tagType)[trainingSize:trainingSize + testingSize]
numberOfSent = 0
numberOfCorrectTags = 0
numberOfWords = 0
confusionmatrix ={}
for tag in countTags.keys():
    if tag not in ["<s>", "</s>"]:
        confusionmatrix[tag] = {}
        for t in countTags.keys():
            if t not in ["<s>", "</s>"]:
                confusionmatrix[tag][t] = 0                   
        confusionmatrix[tag]['Total'] = 0           
    

''' ####################### Viterbi Algorithm ############################## '''
for testingSent in testingSents:
    actualTags = ['<s>'] + [ t for (_, t) in testingTaggedSents[numberOfSent]] + ['</s>']
    actualSent = copy.deepcopy(testingSent)
    ''' Replace word occurring once with 'UNK' '''
    for i in range(len(testingSent)):
        if testingSent[i] not in words:
            testingSent[i] = 'UNK'

    viterbiMatrix = {}
    states = countTags.keys()

    ''' Initialise Viterbi matrix '''
    for tag in states:
        if tag not in viterbiMatrix.keys():
            viterbiMatrix[tag] = {}
        for word in testingSent:
            viterbiMatrix[tag][word] = 0.0

    ''' The predicted POSs will be stored in this list'''
    predictedTags = [0 for x in range(len(testingSent) + 2)]

    ''' initialise step start (first word)'''
    for state in countTags.keys():
        if state not in ['<s>', '</s>']:
            viterbiMatrix[state][testingSent[0]] = getTransitionProbs(state , '<s>') * getEmissionProbs(state, testingSent[0])
            predictedTags[0] = '<s>'
    for t in range(1, len(testingSent)):
        word = testingSent[t]
        wordPrime = testingSent[t - 1]
        backpointer = {}
        backpointer['tag'] = []
        backpointer['value'] = []
        for state in states :

            if state not in ['<s>', '</s>']:
                transitionProb = [getTransitionProbs(state, prevState) * viterbiMatrix[prevState][wordPrime] for prevState in viterbiMatrix.keys()]

                maxTransitionProb = max(transitionProb)
                beta = getEmissionProbs(state, word)
                
                ''' Find which previous state giving maximum prob(transition prob * a) and put it into the list'''
                for prevState in states:
                    if viterbiMatrix[prevState][wordPrime] * getTransitionProbs(state, prevState) == maxTransitionProb:
                        viterbiMatrix[state][word] = maxTransitionProb * beta
                        if viterbiMatrix[state][word] != 0:
                            backpointer['tag'] += [prevState]
                            backpointer['value'] += [maxTransitionProb]
                        break

        '''Find the transition (previous tag -> current tag) that give the maximum transition probability'''
        acutalPrevPOS = backpointer['tag'][backpointer['value'].index(max(backpointer['value']))]
        predictedTags[t] = acutalPrevPOS 

    '''Find POS of the last word(usually is '.')'''
 
    transitionProb = [ viterbiMatrix[prevState][testingSent[-1]] * getTransitionProbs('</s>' , prevState) for prevState in viterbiMatrix.keys()]
    maxTransitionProb = max(transitionProb)
  
    for prevState in states:
        if viterbiMatrix[prevState][testingSent[-1]] * getTransitionProbs('</s>' , prevState) == maxTransitionProb:
            viterbiMatrix['</s>'][testingSent[-1]] = maxTransitionProb
            if viterbiMatrix['</s>'][testingSent[-1]] != 0:
                predictedTags[len(testingSent)] = prevState
                
            break
    ''' POS of the end '''
    predictedTags[-1] = '</s>'

    if numberOfSent % 10 == 0 and numberOfSent is not 0:
        print 'Sentence Number: %d' % (trainingSize + numberOfSent)
    numberOfSent += 1
    ''' Accuracy tracking'''
    for i in range(len(predictedTags)):
        if predictedTags[i] not in ["<s>", "</s>"]:
            if actualTags[i] not in confusionmatrix.keys():
                confusionmatrix[actualTags[i]] =[{t: 0 } for t in countTags.keys()]
                confusionmatrix[actualTags[i]] += [{'Total': 0 }]
            if predictedTags[i] == actualTags[i]:
                numberOfCorrectTags += 1
            numberOfWords += 1
            confusionmatrix[actualTags[i]][predictedTags[i]] += 1
            confusionmatrix[actualTags[i]]['Total'] += 1

  
overAll = " Overall accuracy , correct: %d, total %d, Accuracy: %f %%\n" % (numberOfCorrectTags, numberOfWords, float(numberOfCorrectTags * 100.0 / numberOfWords))
print overAll

''' Write Confusion matrix in CSV file, and print out the matrix on the terminal.'''
with open('Confusion Matrix.csv', 'wb') as csv_file:
    csv_file.write(overAll+"\n")
    csv_file.write(" ,")
    ''' Print Confusion Matrix'''
    print ("%-5s" % "    "),
    for tag in confusionmatrix.keys():
        print ("%-5s" % tag),
        csv_file.write(tag.encode('ascii', 'ignore').decode('ascii')+",")
    csv_file.write("Correct, Total, Accuracy \n")
    print "%-7s" % 'correct',
    print "%-5s" % 'Total',
    print "%9s" % 'Accuracy'
    for tag in confusionmatrix.keys():
           
            print ("%-5s" % tag),
            csv_file.write(tag.encode('ascii', 'ignore').decode('ascii')+",")
            for predTag in confusionmatrix[tag].keys():
                if predTag is not 'Total':
                    csv_file.write(str(confusionmatrix[tag][predTag])+",")
                    print ("%-5d" % confusionmatrix[tag][predTag]),
           
            if confusionmatrix[tag]['Total'] == 0:
                percentage = 0
            else:
                percentage = confusionmatrix[tag][tag] * 100.0 / confusionmatrix[tag]['Total']
            csv_file.write(str(confusionmatrix[tag][tag])+",")
            csv_file.write(str(confusionmatrix[tag]['Total'])+",")
            csv_file.write(str(percentage)+"%"+",")
            print "%-7d" % confusionmatrix[tag][tag],  
            print "%-5d" % confusionmatrix[tag]['Total'],
            print "%9.4f %%" % percentage
            csv_file.write("\n")
