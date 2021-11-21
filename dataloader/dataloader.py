
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
import json
import re
import fasttext
import time
import random
from os import listdir
from os.path import isfile, join


		
# -------------- Cleaning data --------------
def clean_data(statement):
	#Split the basic block text with manually defined delimiters
	X = re.split(' |>|<|,|;|=|\(|\[', statement)
	#print(X)
	
	#Remove empty strings from list
	X = list(filter(None, X))
	#print(X)
	
	#Remove unnecessary strings (in this case, those that have numbers) from list
	i = 0
	while i < len(X):
		if bool(re.search(r'\d', X[i])):
			del(X[i])
			i-=1
		i+=1
		
	#Remove ')' from the beginning of output class
	for i in range(len(X)):
		X[i] = X[i].replace(")", "")
	
	#print(X)
	return X
	

# -------------- Categorize keywords --------------
def categorize_keywords(index, X_block, my_dict=dict()):
	my_dict = dict()
	for i in index:
		for block in X_block:
			#print(i, len(block), block)
			if i >= len(block):
				continue
			
			curr = re.split('\[',block[i])[-1]
			
			if curr not in my_dict.keys():
				my_dict[curr] = len(my_dict) + 1

	return my_dict

# --------------- Utility functions for vectorizing basic blocks ----------------

# Categorize function type like invokespecial, invokevirtual, etc.



# -------------- Phrase2vec for function name --------------

'''
1. The text is assumed to start with lowercase and all subsequent words start with upper case. Exa: readLine
2. Output has a default provision for 5 words in name. This can be modified.
3. Each word has a 10-dim embedding.
'''

#


def phrase2vec(name, model, max_words=5):

   #Convert first character to uppercase
   name = name[0].upper() + name[1:]
   
   #Extract all words starting with uppercase characters
   words = re.findall('[A-Z][^A-Z]*', name)
   
   
   
   #Get word embeddings one word at a time. Need to be in lowercase
   word_embeddings = [model.get_word_vector(x.lower()) for x in words]
   
   #Concatenate individual word embeddings to get phrase embedding
   embeddings = np.zeros(max_words*10)
   
   for i in range(len(word_embeddings)):
	   embeddings[i*10:(i*10 + 10)] = word_embeddings[i]
	   
   #print(embeddings)
   
   return embeddings

# -------------- Combine categorization and phrase2vec to vectorize basic block --------------

#'''
#NLP based processing with options for one-hot encoding and no one-hot



def vectorize(X_block,one_hot,func_type,app_or_pri,func_class,func_io_class,n_words,max_inputs,model,dim):
	
	vec_list = []
	
	for block in X_block:
		#vec = []
		
		# Return 0 vector if basic block is not a function
		if block[0] != 'invokespecial' and block[0] != 'invokevirtual':
			vec = [0] * dim
			vec_list.append(vec)
				
		elif one_hot == False:
			
			func_name = block[3]
			name_embedding = phrase2vec(func_name, max_words=n_words)
			vec = name_embedding.tolist()
			
			block_type = block[0]
			if block_type not in func_type.keys():
				vec.append(0)
			else:
				vec.append(func_type[block_type])
				
			block_app_or_pri = block[1]
			if block_app_or_pri not in app_or_pri.keys():
				vec.append(0)
			else:
				vec.append(app_or_pri[block_app_or_pri])
				
			block_class = block[2]
			if block_class not in func_class.keys():
				vec.append(0)
			else:
				vec.append(func_class[block_class])
			
			#Input vectorization
			
			n = len(block)
			n_inputs = n - 1 - 4
			
			for i in range(n_inputs):
				
				block_input = block[4+i]
				if block_input not in func_io_class.keys():
					vec.append(0)
				else:
					vec.append(func_io_class[block_input])
					
			#Fill in zeros for empty inputs
			k = max_inputs - n_inputs
			
			for i in range(k):
				vec.append(0)
					
			#Output vectorizationn
			
			block_output = block[-1]
			if block_output not in func_io_class.keys():
				vec.append(0)
			else:
				vec.append(func_io_class[block_output])
				
			vec_list.append(vec)
		
		
		
		elif one_hot == True:
			
			#encoded = [0] * (len(func_type) + len(app_or_pri) + len(func_class) + max_inputs * len(func_io_class))
			encoded = [0] * (len(func_type) + len(app_or_pri) + len(func_class) + 2 * len(func_io_class))
			offset = 0
			
			func_name = block[3]
			name_embedding = phrase2vec(func_name,model, max_words=n_words)
			vec = name_embedding.tolist()
			
			block_type = block[0]
			if block_type in func_type.keys():
				key = offset + func_type[block_type] - 1
				encoded[key] = 1
			offset += len(func_type)
				
			block_app_or_pri = block[1]
			if block_app_or_pri in app_or_pri.keys():
				key = offset + app_or_pri[block_app_or_pri] - 1
				encoded[key] = 1
			offset += len(app_or_pri)
				
			block_class = block[2]
			if block_class in func_class.keys():
				key = offset + func_class[block_class] - 1
				encoded[key] = 1
			offset += len(func_class)
			
			#Input vectorization
			
			n = len(block)
			n_inputs = n - 1 - 4
			
			for i in range(n_inputs):
				
				block_input = block[4+i]
				
				if block_input in func_io_class.keys():
					key = offset + i + func_io_class[block_input] - 1
					encoded[key] += 1
			offset += len(func_io_class)
			
			#Output vectorizationn
			block_output = block[-1]
			if block_output in func_io_class.keys():
				key = offset + func_io_class[block_output] - 1
				encoded[key] = 1
				
			vec_list.append(vec + encoded)
		
		
	return vec_list
#'''


	
print("------------- Vectorization complete! -------------\n")



class dataloader:
	'''
	Outputs 
	1. Sequences of dimension batch_size x maxlen x dim
	2. Labels of dimension batch_size x maxlen
	3. Attention masks of dimension batch_size x l
	4. Timestamps of dimension batch_size x l

	'''
	def __init__(self,
				batch_size,
				max_len,
				dir_path,
				device,
				random_seed,
				n_words = 3,
				max_inputs = 4,
				one_hot = True):

		np.random.seed(random_seed)
		torch.manual_seed(random_seed)
		random.seed(random_seed)

		self.device = device
		self.batch_size = batch_size
		self.max_len = max_len
		data_path = dir_path + 'train_data'
		onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]
		seq_array = []
		y = [] #This stores categorical labels for each sequence
		X_block = []
		c = 1
		for exploit in onlyfiles:
			# Opening JSON file
			f = open(data_path + '/' + exploit, )
			# returns JSON object as 
			# a dictionary
			data = json.load(f)
			f.close()
			#print(c)
			#print(exploit)
			for seq in data['SequenceArray']:
				#print("Seqid: " + seq['seqID'])
				curr_seq = []
				for i in range(len(seq['BBSequence'])):
					bb = seq['BBSequence'][i]['BBStatements']
					statements = bb.split('|')
					
					for statement in statements:
						block = clean_data(statement.strip())
						curr_seq.append(block)
						
						if block not in X_block:
							X_block.append(block)
								
				seq_array.append(curr_seq)              
				y.append(c)
			c+=1

		model = fasttext.load_model(dir_path+'dbpedia.bin')
		func_type = categorize_keywords([0], X_block)
		app_or_pri = {'Application': 1, 'Primordial': 2}
		func_class = categorize_keywords([2], X_block)
		func_io_class = categorize_keywords(np.arange(4,len(X_block)+1), X_block)
		if one_hot == True:
			self.dim = n_words * 10 + len(func_type) + len(app_or_pri) + len(func_class) + 2 * len(func_io_class)
		else:
			self.dim = n_words * 10 + 4 + max_inputs

		seq_vector_array = []
		for seq in seq_array:
			seq_vector_array.append(vectorize(seq,one_hot,func_type,app_or_pri,func_class,func_io_class,n_words,max_inputs,model,self.dim))
		
		self.data = seq_vector_array
		self.label = y


	def get_batch(self):
	
		sample_indices = random.sample(range(0,len(self.data)), self.batch_size)
		vectors = np.zeros((self.batch_size,self.max_len,self.dim))
		labels = np.zeros((self.batch_size,self.max_len))
		attn_masks = np.zeros((self.batch_size,self.max_len))
		time_stamps = np.zeros((self.batch_size, self.max_len))

		for i in range(len(sample_indices)):
			index = sample_indices[i]
			v = np.asarray(self.data[index])
			length = len(v)
			curr_attn_mask = np.ones(length)
			curr_time_stamp = np.arange(length)
			
			#Zero-padding
			'''
			zero_vec = [0] * dim
			n_zero_vec = maxlen - length
			for i in range(n_zero_vec):
				v.insert(0,zero_vec) #Potential error
				curr_attn_mask[i] = 0
				curr_time_stamp.insert(0,0)
			'''
			if length < self.max_len:
				#print(v.shape)
				v = np.concatenate((np.zeros((self.max_len-length,self.dim)),v),axis=0)
				curr_attn_mask = np.concatenate((np.zeros(self.max_len - length),curr_attn_mask))
				curr_time_stamp = np.concatenate((np.zeros(self.max_len - length),curr_time_stamp))
			
			
			tmp_label = np.zeros(self.max_len)
			tmp_label[-1] = self.label[index]
			vectors[i] = v
			labels[i] = tmp_label
			attn_masks[i] = curr_attn_mask
			time_stamps[i] = curr_time_stamp
		
		vectors = torch.from_numpy(vectors).to(dtype=torch.float32,device=self.device)
		labels = torch.from_numpy(labels).to(dtype=torch.long,device=self.device)
		attn_masks = torch.from_numpy(attn_masks).to(dtype=torch.long,device=self.device)
		time_stamps = torch.from_numpy(time_stamps).to(dtype=torch.long,device=self.device)
		
		return vectors, labels, attn_masks, time_stamps
