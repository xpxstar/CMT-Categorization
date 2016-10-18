# -*- coding: utf-8 -*-
import sys  
import os 
import numpy as np
#引入Bunch类
from sklearn.datasets.base import Bunch
#引入持久化类
import pickle
from sklearn import feature_extraction  
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.feature_extraction.text import TfidfVectorizer  
from TextPreprocess import TextPreprocess  # 第一个是文件名，第二个是类名
#导入线性核svm算法
from sklearn.svm import LinearSVC

from util import calculate_result


# 配置utf-8输出环境

# 测试语料预处理
testsamp = TextPreprocess()
#testsamp.corpus_path = "test_corpus1_small/"    #原始语料路径
#testsamp.pos_path = "test_corpus1_pos/"       #预处理后语料路径
# 测试语料预处理
#testsamp.preprocess()


testsamp.segment_path = "test_corpus1_segment/"   #分词后语料路径
testsamp.stopword_path = "extra_dict/hlt_stop_words.txt"  #停止词路径
# 为测试语料分词
#testsamp.segment()

# 实际应用中可直接导入分词后测试语料

# 随机选择分好类的测试语料
category = os.listdir(testsamp.segment_path)
random_index = 2 #范围 0 ~ len(category)-1
actual =[]

test_path = testsamp.segment_path+category[random_index]+"/"
test_data=[]
file_list = os.listdir(test_path)
for file_path in file_list:
        file_name = test_path + file_path  # 拼出文件名全路径
        file_obj = open(file_name, "rb")
        test_data.append(file_obj.read())
        actual.append(random_index)
        file_obj.close()

#对测试文本进行tf-idf计算
#从文件导入停用词表
stpwrdlst = testsamp.getStopword(testsamp.stopword_path)

# 导入训练词袋模型
train_set = TextPreprocess()
train_set.wordbag_path = "text_corpus1_wordbag/"
train_set.wordbag_name = "word_bag.data"#词袋文件名
train_set.load_wordbag()
print train_set.wordbag.tdm.shape

# 使用TfidfVectorizer初始化测试文本
#############################################################
# 让两个TfidfVectorizer共享一个vocabulary：
#        vectorizer = TfidfVectorizer(vocabulary=myvocabulary)
#        transformer = TfidfTransformer()
#        return vectorizer.fit_transform(test_data)
#############################################################
fea_test = testsamp.tfidf_value(test_data,stpwrdlst,train_set.wordbag.vocabulary)
print fea_test.shape

#应用linear_svm算法 输入词袋向量和分类标签
#svclf = SVC(kernel = 'linear')   # default with 'rbf'
svclf = LinearSVC(penalty="l1",dual=False, tol=1e-4)
svclf.fit(train_set.wordbag.tdm, train_set.wordbag.label)
# 预测分类结果
predicted = svclf.predict(fea_test)

for file_name,expct_cate in zip(file_list,predicted):
        print "测试语料文件名:",file_name,": 实际类别:",category[random_index],"<-->预测类别:",train_set.wordbag.target_name[expct_cate]

actual=np.array(actual)
#print actual
#print predicted
# 计算分类各种参数指标
calculate_result(actual,predicted)
