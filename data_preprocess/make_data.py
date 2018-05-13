import re
from tqdm import tqdm

#NOTE: if no fullstop in data then won't save

file = "../data/websters_def.txt"
op_file = "../data/one_word.txt"

with open(file,"r") as f:
	lines = f.readlines()

DEF = r"^(Defn:)"
num = r"^(\d)(\.)( )"
alpha = r"^(\()([a-z]{1})(\)(\s))"
BRACKET = r"(\[[^\]]*\])"
POS = r"([anvp]{1}(b)*(d)*(\.)*(\s&\s[anvp]{1}(b)*(d)*)*\.)"

meaning_word_lines = []

word = ""
meaning = ""
meaning_started = False
number = False
prev_def_valid = False

for line in tqdm(lines):
	if re.match(r"^(\n)$",line) != None:
		continue
	if re.match(r"^([A-Z0-9-\s\'\;]+)$",line)!=None:
		actual_word = line.strip("\n")
		word = re.sub(" ","_",line).strip("\n")
		# word = re.sub(";","",line).strip("\n")
		word = re.sub("'","",word).strip("\n")
		meaning = ""
		number = False
	# print('actual word')
	# print(actual_word+'. ')
	# print(line)
	# print()
	if re.match(DEF,line)!=None or re.match(actual_word+'. ',line)!=None:
		if number == True:
			# print('inside number true')
			# print(word)
			meaning = ""
			meaning_word_lines = meaning_word_lines[:-2]
		meaning_started = True
	if re.match(num,line)!=None:
		# print('inside number')
		number = True
		meaning_started = True
		prev_def_valid = False
	if re.match(alpha,line)!=None:
		if number == True and prev_def_valid == False:
			meaning = ""
			meaning_word_lines = meaning_word_lines[:-2]
			prev_def_valid = True
		if meaning != "": #contains a previous definition from an alphabet
			meaning_word_lines.append(meaning.strip())
			meaning_word_lines.append(word)
			meaning = ""
		meaning_started = True
	if meaning_started == True:
		line = re.sub(DEF,"",line).strip("\n")
		line = re.sub("^"+actual_word+". ","",line)
		line = re.sub(num,"",line)
		line = re.sub(alpha,"",line)
		line = re.sub(BRACKET,"",line)
		line = re.sub(POS,"",line)
		line = line.strip(" ")
		if re.search(r"\.",line)!=None:
			meaning += " " + re.search(r"^([^\.]*)(\.)(.*)$",line).groups()[0]
			meaning_word_lines.append(meaning.strip())
			meaning_word_lines.append(word)
			meaning = ""
			meaning_started = False
		else:
			meaning += " " + line
	# print(line)
	# print(meaning_started)
	# print(meaning_word_lines)
	# print()

ind_meaning_word_lines = []
for index,line in enumerate(meaning_word_lines):
	if index%2==0:
		definition = line
	else:
		cur_word = line

		definition_arr = definition.split(';')
		word_arr = cur_word.split(';')

		for _def in definition_arr:
			for word in word_arr:
				ind_meaning_word_lines.append(_def.strip())
				ind_meaning_word_lines.append(word.strip())

with open(op_file,"w") as f:
	f.write("\n".join(ind_meaning_word_lines))
