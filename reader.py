#https://github.com/arksch/NE2vec
#http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
#http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
#http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/


import numpy as np
import re
import sys
import itertools
from collections import Counter


class Reader():


	def __init__(self, category = 'PER'):
		self.category = category
		self.maxNounsInSentence = 0
		self.maxVerbsInSentence = 0
		self.maxAdjectivesInSentence = 0
		return


	def cleanStr(self, string):
	    """
	    Tokenization/string cleaning for all datasets except for SST.
	    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
	    """
	    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
	    string = re.sub(r"\'s", " \'s", string)
	    string = re.sub(r"\'ve", " \'ve", string)
	    string = re.sub(r"n\'t", " n\'t", string)
	    string = re.sub(r"\'re", " \'re", string)
	    string = re.sub(r"\'d", " \'d", string)
	    string = re.sub(r"\'ll", " \'ll", string)
	    string = re.sub(r",", " , ", string)
	    string = re.sub(r"!", " ! ", string)
	    string = re.sub(r"\(", " \( ", string)
	    string = re.sub(r"\)", " \) ", string)
	    string = re.sub(r"\?", " \? ", string)
	    string = re.sub(r"\s{2,}", " ", string)
	    return string.strip()


	def loadDataAndLabels(self, filePath):
	    """
	    Loads MR polarity data from files, splits the data into words and generates labels.
	    Returns split sentences and labels.
	    """
	    # Load data from files
	    
	    samples = list(open(filePath, "r").readlines())
	    sentences = []
	    mainWord = []
	    otherSentences = []
	    sentence = []
	    y = []
	    index = 0
	    for row in samples:
	    	columns = row.split(" ")

	    
	    	if columns[0] in ['\n']:
	    		if y and sentence:
	    			for item in y:
	    				sentences.append(sentence)
	    				mainWord.append(y)
	    		elif sentence:
	    			otherSentences.append(sentence)

	    		index += 1
	    		sentence = []
	    		y = []
	    		continue
	    	
	    	if (len(columns) == 1) or (columns[1] not in ['NNP', 'NNPS', 'NN', 'NP', 'JJ', 'NNS', 'PRP', 'VBZ', 'VB', 'VBD', 'VBP', 'VP']):
	    		continue

	    	sentence.append(columns[0])

	    	if self.category in columns[3]:
	    		y.append(columns[0])

	    if y and sentence:
	    	for item in y:
	    		sentences.append(sentence)
	    		mainWord.append(y)

	    del samples
	    samples = []
	    output = []

	    for sentence in sentences:
	    	samples.append(' '.join(sentence))
	    	output.append([1, 0])

	    for sentence in otherSentences:
	    	samples.append(' '.join(sentence))
	    	output.append([0, 1])
	    
	    '''		
	    negative_examples = list(open(negative_data_file, "r").readlines())
	    negative_examples = [s.strip() for s in negative_examples]
	    # Split by words
	    #x_text = positive_examples + negative_examples
	    x_text = [self.cleanStr(sent) for sent in negative_examples]
	    #print(x_text)
	    # Generate labels
	    #positive_labels = [[0, 1] for _ in positive_examples]
	    negative_labels = [[1, 0] for _ in negative_examples]
	    #y = np.concatenate([positive_labels, negative_labels], 0)
	    y = []
	    '''
	    #print(np.concatenate([output], 0))
	    return [samples, np.concatenate([output], 0)]


	def batch_iter(self, data, batch_size, num_epochs, data_size, shuffle=True):
	    """
	    Generates a batch iterator for a dataset.
	    """
	    print(data_size)
	    #print(batch_size)
	    #print('---------------------------------------')
	    #print(np.array(data))
	    #print('---------------------------------------')
	    data = np.array(data)
	    #data_size = len(data)
	    num_batches_per_epoch = int((data_size-1)/batch_size) + 1
	    print(num_batches_per_epoch)
	    
	    for epoch in range(num_epochs):
	        # Shuffle the data at each epoch
	        if shuffle:
	            shuffle_indices = np.random.permutation(np.arange(data_size))
	            shuffled_data = data[shuffle_indices]
	        else:
	            shuffled_data = data
	        for batch_num in range(num_batches_per_epoch):
	            start_index = batch_num * batch_size
	            end_index = min((batch_num + 1) * batch_size, data_size)
	            yield shuffled_data[start_index:end_index]