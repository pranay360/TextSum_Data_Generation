# TextSummarization
Generating Dataset for Google's Text Summarization Code by Xin Pan and Peter Liu 
Repository Link: https://github.com/tensorflow/models/tree/master/textsum

Dataset can be obtained here: CNN stories http://cs.nyu.edu/~kcho/DMQA/

Working:
The valid data format requires article and abstract key for the TextSum algorithm to train and decode.
Both articles and abstracts are tagged for sentence, paragraph and document start and end.
abstract is extracted using all @highlights in data.
Vocabulary with 100000 words include UNK and PAD tokens are generated.

Usage:
CNN Stories Dataset should be in %pwd%/stories
run mkdir data in the present working directory
run python convertdata.py

The script will also work for Dailymail stories.

