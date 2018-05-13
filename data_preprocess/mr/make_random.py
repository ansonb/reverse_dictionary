import random
import json
import re
import tqdm

data_dir = '../../data/mr/one_word_mr.txt'
save_path = '../../data/mr/one_word_noise_sample_mr.json'
save_out_words_path = '../../data/mr/embedding_arr_names.csv'

_dict = {}
group_size = 3
max_num_defs = 10
max_num_defs_large = 100
#TODO: include a common helper file
def preprocess_data(text):
	# text = clean_data(text)
	text = text.lower()
	text = text.strip()
	return text

_WORD_SPLIT = re.compile("([.,!?\"':;)(])")
def basic_tokenizer(sentence):
  """Very basic tokenizer: split the sentence into a list of tokens."""
  words = []
  for space_separated_fragment in sentence.strip().split():
    words.extend(re.split(_WORD_SPLIT, space_separated_fragment))
  return [w for w in words if w]

def shuffle(words_to_shuffle, index):
	if index>=len(words_to_shuffle):
		return words_to_shuffle
	
	shuffled_word_index = random.randint(index,len(words_to_shuffle)-1)
	selected_word = words_to_shuffle[shuffled_word_index]
	words_to_shuffle[shuffled_word_index] = words_to_shuffle[index]
	words_to_shuffle[index] = selected_word

	return shuffle(words_to_shuffle,index+1)

def shuffle_all(words_to_shuffle, index):
	if index>=len(words_to_shuffle):
		return words_to_shuffle
	
	shuffled_word_index = random.randint(index,len(words_to_shuffle)-1)
	selected_word = words_to_shuffle[shuffled_word_index]
	words_to_shuffle[shuffled_word_index] = words_to_shuffle[index]
	words_to_shuffle[index] = selected_word

	return shuffle_all(words_to_shuffle,index+1)

def getDef(text, _round):
	words = basic_tokenizer(text)
	begin = (_round*group_size)%len(words)
	end = min(begin+group_size,len(words))
	words_to_shuffle = words[begin:end]
	shuffled_words = shuffle(words_to_shuffle,0)

	words[begin:end] = shuffled_words
	shuffled_text = ' '.join(words).strip()
	return shuffled_text

with open(data_dir,'r') as f:
	lines = f.readlines()

	for index,line in tqdm.tqdm(enumerate(lines),total=len(lines)):
		line = preprocess_data(line)
		if index%2 == 0:
			cur_def = line
		else:
			if _dict.get(line,-1) == -1:
				_dict[line] = []
			_dict[line].append(cur_def)

word_arr = []
for word,definitions in tqdm.tqdm(_dict.items(),total=len(_dict.items())):
	def_arr = definitions
	# def_arr.append(definition)

	for definition in definitions:
		while len(def_arr)<=max_num_defs:
			new_def = getDef(definition,len(def_arr))
			def_arr.append(new_def)

		while len(def_arr)<=max_num_defs_large+max_num_defs+1:
			new_def_words_arr = shuffle_all(basic_tokenizer(definition),0)
			def_arr.append(' '.join(new_def_words_arr).strip())

		def_arr.append(word)

	_dict[word] = def_arr

	word_arr.append(word)

with open(save_path,'w') as f:
	json.dump(_dict,f,indent=4)

with open(save_out_words_path,'w') as f:
	json.dump(word_arr,f,indent=4)