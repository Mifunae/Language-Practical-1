usage: HMM.py [-h] [-u] [-gt] corpus trainingSize testingSize

HMM tagging.

positional arguments:
  corpus        A string which is the name of the corpus. Which could be either Brown, Conll2000, Conll2002, Alpino, Floresta, or Treebank

  trainingSize  an integer for the trainingSize. 
		The program will start training until reaching certain amount(trainingSize) of sentences in selected corpus. For example, if the trainingSize= 500, 
		the sentences position 0 to 499 in corpus will be usedfor training.

  testingSize   an integer for the testingSize. The amount of testing sentences in the selected corpus which will start testing at the position after the trainingSize. 
		For example, if the trainingSize = 500, testingSize is 300, the sentences position 500 to 799 in corpus will be used for testing

optional arguments:
  -h, --help    show this help message and exit

  -u            Use the universal tag, if available.

  -gt           Use Good-Turing smoothing technique in computing transition probability,unless the Laplace technique
                will be used instead.

Reminder
1. Sometimes there is an error happens when the training size is not high enough, and I couldn't fix it. So, when there is an error please increase number of training size.
2. Runing this program on Floresta corpus will take a long time in testing