import os
import re
import numpy as np
from tensorflow.core.example import example_pb2
from nltk.tokenize import sent_tokenize
import struct
import collections
counter = collections.Counter()

files=os.listdir("stories/")
n_files=len(files)
tr_r=0.8
v_r=0.12

print "Total Files:", n_files
print
train=files[:int(n_files*0.8)]
validation=files[len(train):len(train)+int(n_files*0.12)]
test=files[len(train)+len(validation):]

def convert_text2bin(docs, writer):
    for i, fi in enumerate(docs):
        with open("stories/"+fi,'r') as f:
            wholetext=f.read().decode('utf8').lower()
            wholetext=re.sub(r'[^\x00-\x7F]+','', wholetext)
            data=wholetext.split("@highlight")
            news=data[0]
            highlights=(". ".join([h.replace('\n\n','') for h in data[1:]])+".").strip()
            news=(" ".join(news.split('\n\n'))).strip()        
            sentences = sent_tokenize(news)
            news = '<d> <p> ' + ' '.join(['<s> ' + sentence + ' </s>' for sentence in sentences]) + ' </p> </d>'
            sentences = sent_tokenize(highlights)
            highlights = '<d> <p> ' + ' '.join(['<s> ' + sentence + ' </s>' for sentence in sentences]) + ' </p> </d>'
            words = (news+" "+highlights).split()
            counter.update(words)
            tf_example = example_pb2.Example()
            tf_example.features.feature['article'].bytes_list.value.extend([news.encode('utf-8')])
            tf_example.features.feature['abstract'].bytes_list.value.extend([highlights.encode('utf-8')])
            tf_example_str = tf_example.SerializeToString()
            str_len = len(tf_example_str)
            writer.write(struct.pack('q', str_len))
            writer.write(struct.pack('%ds' % str_len, tf_example_str))
            if i%3000==0:
                print int((float(i)/ len(docs))*100), "%"
    print (float(len(docs))/ len(docs))*100, "%...." "converted\n\n"


print "Generating Training Data\n"
with open('data/trainCNN.bin', 'wb') as writer:
    convert_text2bin(train,writer)
print "Generating Validation Data\n"
with open('data/validationCNN.bin', 'wb') as writer:
    convert_text2bin(validation,writer)
print "Generating Testing Data\n"
with open('data/testCNN.bin', 'wb') as writer:
    convert_text2bin(test,writer)

print "Data Generated"
print "Train:\t\t",len(train)
print "Validation:\t",len(validation)
print "Test:\t\t",len(test)
print

print "Generating Vocabulary"
mc=counter.most_common(100000-2)
with open("data/vocabCNN", 'w') as writer:
    for word, count in mc:
        writer.write(word + ' ' + str(count) + '\n')
    writer.write('<UNK> 0\n')
    writer.write('<PAD> 0\n')
print "Vocab Generated with total no. of words:",len(mc)

print "\n\nData Generation Finished..."
