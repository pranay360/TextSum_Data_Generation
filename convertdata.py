'''CNN and DM data should be present in ./cnn/stories and ./dailymail/stories'''
import os
import re
import sys
import struct
import numpy as np
import collections
from nltk.tokenize import sent_tokenize
from tensorflow.core.example import example_pb2

counter = collections.Counter()
temp=0
#train, test, validation split
tr_r=0.85
v_r=0.08
if len(sys.argv)==1 or len(sys.argv)>2:
    print "Incorrect Usage"
    print "Usage: python convertdata.py --CNN or --DM or --both"
    exit()

if sys.argv[1]=="--CNN" or sys.argv[1]=="--both":
    print 'Generating CNN data....'
    print
    files=os.listdir("cnn/stories/")
    n_files=len(files)

    print "Total Files:", n_files
    print
    train=files[:int(n_files*0.8)]
    validation=files[len(train):len(train)+int(n_files*0.12)]
    test=files[len(train)+len(validation):]

    def convert_text2bin1(docs, writer):
        global counter
        for i, fi in enumerate(docs):
            with open("cnn/stories/"+fi,'r') as f:
                wholetext=f.read().decode('utf8').lower()
                wholetext=re.sub(r'[^\x00-\x7F]+','', wholetext)
                wholetext=re.sub(r"(\s?[\']\s+|\s+[\']\s?)"," ' ", wholetext)
                wholetext=re.sub(r'(\s?[\"]\s+|\s+[\"]\s?)',' " ', wholetext)
                wholetext=re.sub(r"(\'[s]\s+)"," 's ", wholetext)
                wholetext=wholetext.replace("."," . ")
                wholetext=wholetext.replace(","," , ")
                wholetext=wholetext.replace('-',' - ')
                wholetext=wholetext.replace('?',' ? ')
                wholetext=wholetext.replace('(','( ')
                wholetext=wholetext.replace(')',' )')
                data=wholetext.split("@highlight")
                news=data[0]
                highlights=data[1].replace('\n\n','')
                news=(" ".join(news.split('\n\n'))).strip()
                sentences = sent_tokenize(news)
                news = '<d> <p> ' + ' '.join(['<s> ' + sentence + ' </s>' for sentence in sentences]) + ' </p> </d>'
                highlights = '<d> <p> <s> ' + highlights + ' </s> </p> </d>'
                words = (news+" "+highlights).split()
                counter.update(words)
                tf_example = example_pb2.Example()
                tf_example.features.feature['article'].bytes_list.value.extend([(' '.join(news.split())).encode('utf-8')])
                tf_example.features.feature['abstract'].bytes_list.value.extend([(' '.join(highlights.split())).encode('utf-8')])
                tf_example_str = tf_example.SerializeToString()
                str_len = len(tf_example_str)
                writer.write(struct.pack('q', str_len))
                writer.write(struct.pack('%ds' % str_len, tf_example_str))
                if i%3000==0:
                    print int((float(i)/ len(docs))*100), "%"
        print (float(len(docs))/ len(docs))*100, "%...." "converted\n\n"

    print "Generating Training Data\n"
    with open('data/trainCNN.bin', 'wb') as writer:
        convert_text2bin1(train,writer)
    print "Generating Validation Data\n"
    with open('data/validationCNN.bin', 'wb') as writer:
        convert_text2bin1(validation,writer)
    print "Generating Testing Data\n"
    with open('data/testCNN.bin', 'wb') as writer:
        convert_text2bin1(test,writer)

    ntrain=len(train)
    nval=len(validation)
    ntest=len(test)
    print "CNN Data Generated"
    print "Train:\t\t",len(train)
    print "Validation:\t",len(validation)
    print "Test:\t\t",len(test)
    print
    print
    temp=n_files

if sys.argv[1]=="--DM" or sys.argv[1]=="--both":
    print "Generating DailyMail data...."
    print
    files=os.listdir("dailymail/stories/")
    n_files=len(files)


    print "Total Files:", n_files
    print
    train=files[:int(n_files*0.8)]
    validation=files[len(train):len(train)+int(n_files*0.12)]
    test=files[len(train)+len(validation):]

    def convert_text2bin2(docs, writer):
        global counter
        for i, fi in enumerate(docs):
            with open("dailymail/stories/"+fi,'r') as f:
                wholetext=f.read().decode('utf8').lower()
                wholetext=re.sub(r'[^\x00-\x7F]+','', wholetext)
                wholetext=re.sub(r"(\s?[\']\s+|\s+[\']\s?)"," ' ", wholetext)
                wholetext=re.sub(r'(\s?[\"]\s+|\s+[\"]\s?)',' " ', wholetext)
                wholetext=re.sub(r"(\'[s]\s+)"," 's ", wholetext)
                wholetext=wholetext.replace("."," . ")
                wholetext=wholetext.replace(","," , ")
                wholetext=wholetext.replace('-',' - ')
                wholetext=wholetext.replace('?',' ? ')
                wholetext=wholetext.replace('(','( ')
                wholetext=wholetext.replace(')',' )')
                data=wholetext.split("@highlight")
                news=data[0]
                try:
                    news=news.split("updated:")[1]
                    news=news[news.find('20')+4:]
                except:
                    None
                news=(" ".join(news.split('\n'))).strip()
                highlights=data[1].replace('\n\n','')
                news=(" ".join(news.split('\n\n'))).strip()
                sentences = sent_tokenize(news)
                news = '<d> <p> ' + ' '.join(['<s> ' + sentence + ' </s>' for sentence in sentences]) + ' </p> </d>'
                highlights = '<d> <p> <s> ' + highlights + ' </s> </p> </d>'
                words = (news+" "+highlights).split()
                counter.update(words)
                tf_example = example_pb2.Example()
                tf_example.features.feature['article'].bytes_list.value.extend([(' '.join(news.split())).encode('utf-8')])
                tf_example.features.feature['abstract'].bytes_list.value.extend([(' '.join(highlights.split())).encode('utf-8')])
                tf_example_str = tf_example.SerializeToString()
                str_len = len(tf_example_str)
                writer.write(struct.pack('q', str_len))
                writer.write(struct.pack('%ds' % str_len, tf_example_str))
                if i%3000==0:
                    print int((float(i)/ len(docs))*100), "%"
        print (float(len(docs))/ len(docs))*100, "%...." "converted\n\n"


    print "Generating Training Data\n"
    with open('data/trainDM.bin', 'wb') as writer:
        convert_text2bin2(train,writer)
    print "Generating Validation Data\n"
    with open('data/validationDM.bin', 'wb') as writer:
        convert_text2bin2(validation,writer)
    print "Generating Testing Data\n"
    with open('data/testDM.bin', 'wb') as writer:
        convert_text2bin2(test,writer)

    print "DailyMail Data Generated"
    print "Train:\t\t",len(train)
    print "Validation:\t",len(validation)
    print "Test:\t\t",len(test)
    print


print "Generating Vocabulary"

mc=counter.most_common(200000-2)
with open("data/vocab", 'w') as writer:
    for word, count in mc:
        writer.write(word + ' ' + str(count) + '\n')
    writer.write('<UNK> 0\n')
    writer.write('<PAD> 0\n')


print "\n\nData Generation Finished...\n\n"
if sys.argv[1]=="--CNN":
    print "CNN Generated"
    temp=0
elif sys.argv[1]=="--DM":
    print "DM Generated"
else:
    print "CNN+DailyMail Data Generated"

print "Total Records",temp+n_files
print "Total Train",ntrain+len(train)
print "Total Validation",nval+len(validation)
print "Total Test",ntest+len(test)
print "Vocab Generated with total no. of words:",len(mc)+2
