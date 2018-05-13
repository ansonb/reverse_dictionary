import os

dataDirPrefix = '../data/sample'


# parsedJsonInputFile = os.path.join(dataDirPrefix,'one_word_parsed_sample.json')
# parsedJsonInputFileTest = os.path.join(dataDirPrefix,'one_word_parsed_test_sample.json')
# dataDir = os.path.join(dataDirPrefix,'one_word_mr.txt')
# dataNoise = os.path.join(dataDirPrefix,'one_word_noise_sample_mr.json')
# embedding_out_names_path = os.path.join(dataDirPrefix,'embedding_arr_names.csv')
parsedJsonInputFile = os.path.join(dataDirPrefix,'one_word_parsed_sample.json')
parsedJsonInputFileTest = os.path.join(dataDirPrefix,'one_word_parsed_test_sample.json')
dataDir = os.path.join(dataDirPrefix,'one_word.txt')
dataNoise = os.path.join(dataDirPrefix,'one_word_noise.json')
embedding_out_names_path = os.path.join(dataDirPrefix,'embedding_arr_names.csv')

embedding_size = 32

num_epochs = 2000
save_every = 100
eval_after = 100
# checkpoint_dir = "../model/mr_dict/lstm"
# checkpoint_prefix = os.path.join(checkpoint_dir, "model")
# parsed_model_checkpoint_dir = "../model/mr_dict/rnn"
# parsed_model_path_prefix = os.path.join(parsed_model_checkpoint_dir, "model")
# parsed_model_checkpoint_dir_2 = "../model/mr_dict/rnn_2"
# parsed_model_path_prefix_2 = os.path.join(parsed_model_checkpoint_dir_2, "model")
checkpoint_dir = "../model/entire"
checkpoint_prefix = os.path.join(checkpoint_dir, "model")
parsed_model_checkpoint_dir = "../model/entire_rnn"
parsed_model_path_prefix = os.path.join(parsed_model_checkpoint_dir, "model")
parsed_model_checkpoint_dir_2 = "../model/entire_rnn_2"
parsed_model_path_prefix_2 = os.path.join(parsed_model_checkpoint_dir_2, "model")
restore = False

#mr
# max_seq_len
# 99
# number of classes
# 7779

#sample
# seq_len = max_seq_len = 62
# NUM_CLASSES = 85

#TODO: save in an output config
seq_len = max_seq_len = 66
NUM_CLASSES = 144

out_embedding_arr_path = os.path.join(dataDirPrefix,'embedding_arr.csv')
out_names_path = os.path.join(dataDirPrefix,'embedding_arr_names.csv')

out_embedding_arr_path_parsed = os.path.join(dataDirPrefix,'embedding_arr_parsed.csv')
out_names_path_parsed = os.path.join(dataDirPrefix,'embedding_arr_names_parsed.csv')

data_json_path = os.path.join(dataDirPrefix,'one_word_noise_sample_extra_large.json')

batch_size = 64
batch_size_parsed = 10
#0 for top 1
#1 for top 3
measure = 0

testDataPath = os.path.join(dataDirPrefix,'testData.txt')
wordTestDataPath = os.path.join(dataDirPrefix,'wordTestData.txt')

dictionaryFile = os.path.join(dataDirPrefix,'dictionary_end2end.json')

dictionaryPath = os.path.join(dataDirPrefix,'dictionary_end2end.json')
dictionaryPathParsed = os.path.join(dataDirPrefix,'dictionary_end2end_parsed.json')
dictionaryPathParsed_2 = os.path.join(dataDirPrefix,'dictionary_end2end_parsed_2.json')
posDictPath = os.path.join(dataDirPrefix,'dictionary_pos_parsed.json')

load_parsed_data = False
parsed_x_batch_path = os.path.join(dataDirPrefix,'x_train_parsed.json')
parsed_y_batch_path = os.path.join(dataDirPrefix,'y_train_parsed.json')
level_arr = [5, 7, 5, 7, 4, 4, 2, 1, 2, 5, 3, 3, 1]

MAX_NUM_BUCKETS_TO_TRAIN = 2


#mr dataset
mr_data_path_train = '../data/baselines/mr/train.tsv'
mr_data_path_test = '../data/baselines/mr/test.tsv'
mrDictionaryPath = '../data/mr/dictionary.json'

mr_num_epochs = 2000
mr_save_every = 100
mr_eval_after = 100
mr_checkpoint_dir = "../model/mr/lstm"
mr_checkpoint_prefix = os.path.join(mr_checkpoint_dir, "model")
mr_checkpoint_dir_pretrained = "../model/mr/lstm"
mr_checkpoint_prefix_pretrained = os.path.join(mr_checkpoint_dir_pretrained, "model")
mr_checkpoint_dir_pretrained_finetuned = "../model/mr/lstm"
mr_checkpoint_prefix_pretrained_finetuned = os.path.join(mr_checkpoint_dir_pretrained_finetuned, "model")
mr_restore = True
mr_save_test_path = 'output/mr/submission.csv'

mr_pretrained = True
mr_fine_tune = False