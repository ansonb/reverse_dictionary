import json

dictionary_path = '../../data/one_word.txt'
vocab_dict = '../../data/mr/dictionary.json'
out_path_one_word_mr = '../../data/mr/one_word_mr.txt'

with open(dictionary_path,'r') as f:
	lines = f.readlines()

with open(vocab_dict,'r') as f:
	dictionary = json.load(f)

print(dictionary)

one_word_mr = []
for index,line in enumerate(lines):
	if index%2==1:
		if dictionary.get(line.strip().lower(),-1)!=-1:
			print('inside -1')
			def_word = line

			one_word_mr.append(definition)
			one_word_mr.append(def_word)
	else:
		definition = line

with open(out_path_one_word_mr,'w') as f:
	f.write(''.join(one_word_mr))