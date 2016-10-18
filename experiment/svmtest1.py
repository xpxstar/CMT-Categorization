# coding=utf-8  
'''
Created on 2016年6月27日

@author: admin
'''
from experiment.bayestest2 import testpath
'''
sklearn里面的TF-IDF主要用到了两个函数：CountVectorizer()和TfidfTransformer()。
    CountVectorizer是通过fit_transform函数将文本中的词语转换为词频矩阵。
    矩阵元素weight[i][j] 表示j词在第i个文本下的词频，即各个词语出现的次数。
    通过get_feature_names()可看到所有文本的关键字，通过toarray()可看到词频矩阵的结果。
    TfidfTransformer也有个fit_transform函数，它的作用是计算tf-idf值。
'''

import codecs
import csv
import os 
import time          

from sklearn.cross_validation import cross_val_score, KFold
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.metrics import precision_recall_fscore_support as score
from  sklearn.svm import SVC

from Predict import Predict
import numpy as np
from util import calculate_result


path='F:/paper/desp/4comp/'
# testpath = 'F:/paper/desp/test/'
testpath = 'F:/paper/casestudy/test httpd.txt'
def tain_classify(filename):
    #如果是excel另存为csv 则需要修改读取方式r为wU
    predict = Predict()
    testfile = file(testpath, 'rU')
    reader2 = csv.reader(testfile)
    predict.test_data=[]
    predict.test_cate=[]
    #读取预料 一行预料为一个文档
    tt=0
    for line in reader2:
        predict.test_data.append(line[0])
#         predict.test_cate.append(line[1])
#         tt+=(int)(line[1])
    testfile.close()
#     if( tt==0 or tt==len(predict.test_data)):
#         print 'None'
#         os.remove(os.path.join(testpath, filename))
#         return None
    
    if not os.path.isfile(path+filename):
        return None
    datafile = file(path+filename, 'rU')
    reader = csv.reader(datafile)
    predict.train_data =[]
    predict.train_cate=[]
    #读取预料 一行预料为一个文档
    for line in reader:
        predict.train_data.append(line[0])
        predict.train_cate.append(line[1])
    datafile.close()
    
        
    #print predict.train_data
    time.sleep(1)
    
    #将文本中的词语转换为词频矩阵 矩阵元素a[i][j] 表示j词在i类文本下的词频
    vectorizer = CountVectorizer(binary = False, decode_error = 'ignore',stop_words = 'english')

    #该类会统计每个词语的tf-idf权值
    transformer = TfidfTransformer()
#     tfidf_data = transformer.fit_transform(vectorizer.fit_transform(data))
    #第一个fit_transform是计算tf-idf 第二个fit_transform是将文本转为词频矩阵
    tfidf_train = transformer.fit_transform(vectorizer.fit_transform(predict.train_data))
    
    vectorizer_test = CountVectorizer(vocabulary=vectorizer.vocabulary_,decode_error = 'ignore')
    tfidf_test = transformer.fit_transform(vectorizer_test.fit_transform(predict.test_data))
    #获取词袋模型中的所有词语  
#     print 'Size of fea_train:' + repr(tfidf_train.shape) 
#     print 'Size of fea_test:' + repr(tfidf_test.shape) 
#     word = vectorizer.get_feature_names()
# 
#     #将tf-idf矩阵抽取出来，元素w[i][j]表示j词在i类文本中的tf-idf权重
#     weight = tfidf.toarray()
# 
#     resName = "BaiduTfidf_Result.txt"
#     result = codecs.open(resName, 'w', 'utf-8')
#     for j in range(len(word)):
#         result.write(word[j] + ' ')
#     result.write('\r\n\r\n')
# 
#     #打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重  
#     for i in range(len(weight)):
#         print u"-------这里输出第",i,u"类文本的词语tf-idf权重------"  
#         for j in range(len(word)):
#             result.write(str(weight[i][j]) + ' ')
#         result.write('\r\n\r\n')
# 
#     result.close()
    svclf = SVC(kernel = 'linear')
    svclf.fit(tfidf_train,predict.train_cate)
    pred = svclf.predict(tfidf_test)
#     outputf = codecs.open("filename",'w','utf-8')#open for 'w'riting
#     precision, recall, fscore, support = score(predict.test_cate, pred,average='binary',pos_label='1')
#     cores = cross_val_score(svclf,data,cate,cv=5)
#     svclf.fit(tfidf_train,predict.train_cate)
#     joblib.dump(svclf, 'database.m')
#    svclf = joblib.load('database.m')
#     pred = svclf.predict(tfidf_test)
    return '{}\t{}\t{}\n'.format(filename,pred[0],pred[1])
if __name__ == "__main__":
    filelist = os.listdir(path)
    outputf = codecs.open('test_httpd.txt','w','utf-8')#open for 'w'riting
    for data in filelist:
#         print file;
        print data;
        outputf.write(tain_classify(data))
    outputf.close()
            
