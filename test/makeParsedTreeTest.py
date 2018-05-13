import subprocess
import tqdm
import json

import sys
sys.path.append('./../')
import config

file = config.testDataPath

def getDepTree(parsed_arr,index,level):
	if index>=len(parsed_arr):
		return []

	cur_level = parsed_arr[index][3]
	# print('cur_level')
	# print(cur_level)
	# print('level')
	# print(level)
	# print('word')
	# print(parsed_arr[index][0])
	if index>0:
		if cur_level <= level:
			return []

		level_arr = []
		
		for i,item in enumerate(parsed_arr[index:]):
			if item[3] < cur_level:
				break
			elif cur_level == item[3]:
				level_arr.append({
					'word': item[0],
					'pos': item[1],
					'dep_tag': item[2],
					'next': getDepTree(parsed_arr,index+i+1,cur_level)
				})


		return level_arr
	else:
		obj = {
			'word': parsed_arr[index][0],
			'pos': parsed_arr[index][1],
			'dep_tag': parsed_arr[index][2],
			'next': getDepTree(parsed_arr,index+1,cur_level)
		}

		return obj

##test
# with open('../ml/syntaxnet/outputs/output_3.txt','r') as f:
# 	lines = f.readlines()
# 	parsed_arr = []
# 	for line in lines[2:]:
# 		index = line.find('+-- ')
# 		if index>0:
# 			index += 4
# 		else: 
# 			index = 0
# 		level = index//4
# 		line = line[index:].strip()
# 		word,pos,dep_tag = line.split()
# 		parsed_arr.append([word,pos,dep_tag,level])

# 	print('parsed_arr')
# 	print(parsed_arr)
# 	print(getDepTree(parsed_arr,0,0))

def getDefData(dataLines):
	def_data = []
	for index,line in enumerate(dataLines):
		if index%2 == 1:
			def_word = line.strip()
			def_data.append([def_word,def_arr])
		else:
			def_arr = [line.strip()]


	return def_data

with open(file, 'r') as f:
	data = f.readlines()
	def_data = getDefData(data)

	parse_tree_arr = []

	for def_word,def_arr in def_data:

		print('========================================= Processing word {} ========================================='.format(def_word))

		full_text = def_arr[0]

		line_arr = full_text.split(';')

		parse_tree_obj_single_sent = {}
		parse_tree_obj_single_sent['sent'] = full_text
		parse_tree_obj_single_sent['def_word'] = def_word
		parse_tree_obj_single_sent['dep_tree'] = []

		for text in line_arr:
			subprocess.call([    
			'echo "' + text + '" | docker run --rm -i brianlow/syntaxnet-docker > syntaxnet_output_test.txt'
			], shell = True)
			

			with open('syntaxnet_output_test.txt','r') as f1:
				parsed_output_lines = f1.readlines()

				if len(parsed_output_lines)>0:
					parse_tree_obj = {}
					parse_tree_obj['sent'] = parsed_output_lines[0]

					parsed_arr = []

					for line in parsed_output_lines[2:]:
						index = line.find('+-- ')
						if index>0:
							index += 4
						else: 
							index = 0
						level = index//4
						line = line[index:].strip()
						
						# print('line')
						# print(line)

						word,pos,dep_tag = line.split()
						parsed_arr.append([word,pos,dep_tag,level])
						# print(word)

					# print('---------------------')
					# print(word)
					parse_tree_obj['dep_tree'] = getDepTree(parsed_arr,0,0)

					# parse_tree_obj['word'] = def_word

					parse_tree_obj_single_sent['dep_tree'].append(parse_tree_obj)
					parse_tree_arr.append(parse_tree_obj_single_sent)
				else:
					print('*************************************** Unable to process ***************************************')

			
		

with open('../data/sample/one_word_parsed_test_sample.json','w') as f:
	json.dump(parse_tree_arr,f,indent=4)

