import numpy as np
from collections import defaultdict
from re import compile,findall,split
import random
with open("ratings.txt") as f:
    ratings = f.readlines()
 
n = random.sample(range(0, len(ratings)), int(len(ratings) * 0.7))
 
print(n)
 
rating_train = []
rating_test = []
 
for i in range(len(ratings)):
    if i in n:
        rating_train.append(ratings[i])
    else:
        rating_test.append(ratings[i])
 
 
filename1 = 'trainset.txt'
with open(filename1, 'w') as f:
    for i in rating_train:
        f.write(i)
 
filename2 = 'testset.txt'
with open(filename2, 'w') as f:
    for i in rating_test:
        f.write(i)
        
#减少数据集样本
'''
rating_list = []
for i in range(int(float(len(trainingData))*float(0.9))):
    for item in trainingData[str(i)]:
        rating_list.append(str(i)+' '+str(item)+' '+str(trainingData[str(i)][item])+'\n')
filename = 'rating10percent.txt'
with open(filename,'w') as f:
    for i in range(len(rating_list)):
        f.write(rating_list[i])
'''