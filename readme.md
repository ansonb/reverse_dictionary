This is a reverse dictionary application using the Webster's unabridged dictionary based on
1) LSTM
2) RNN (Recursive Neural Networks)

Installation:
==============
1)Python 3
Dependencies:
Tensorflow

2) Syntaxnet

3) Docker (optional; for running syntaxnet)

Preparing data
===============
1) Make a folder named data in the root directory
2) Place the websters dictionary from this link: https://raw.githubusercontent.com/matthewreagan/WebstersEnglishDictionary/master/WebstersEnglishDictionary.txt into the data folder
3) Rename the file as websters_def.txt
4) From data_preprocess run make_data.py

For LSTM:
To randomise and augment the data run make_random.py in the data_preprocess folder

For RNN:
To make the parsed trees run make_parse_tree.py (config.py in the root directory contains the input and output file locations)

Training
===============
1)LSTM
In ml folder run train_log_loss.py

2) RNN
In ml folder run train_recursive_nn.py

3) RNN 2
In ml folder run train_recursive_nn_2.py

config.py contains the parameters and the model and data file paths

Inference/Test
==============
1)LSTM
In test folder run test_end2end.py

2) RNN
In ml folder run test_parsed.py

3) RNN 2
In ml folder run test_parsed_2.py