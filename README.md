# TextSummarization
Generating Dataset for Google's Text Summarization Code by Xin Pan and Peter Liu 

Repository Link: https://github.com/tensorflow/models/tree/master/research/textsum

Dataset can be obtained here: CNN and DailyMail stories http://cs.nyu.edu/~kcho/DMQA/


Working:

The valid data format requires article and abstract key for the TextSum algorithm to train and decode.

Both articles and abstracts are tagged for sentence, paragraph and document start and end.

abstract is extracted using all @highlights in data.

Vocabulary with 200000 words include UNK and PAD tokens are generated.



Usage:

CNN and DailyMail data should be present in %pwd%/cnn/stories and %pwd%/dailymail/stories

run mkdir data in the present working directory

You can opt for generating both Datasets or one of them using the following arguments-

run python convertdata.py --both or --CNN or --DM

