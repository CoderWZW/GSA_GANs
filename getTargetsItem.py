# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 17:08:30 2018

@author: WZW
"""
import numpy as np
from collections import defaultdict
from re import compile,findall,split
import random
import numpy as np
with open("dataset/Epinions/testset.txt") as f:
    ratings = f.readlines()


trainingData = defaultdict(dict)
trainingSet_i = defaultdict(dict)
threshold_rating = 3
threshold_frequency = 5
targets_number = 20

item_times = {}
for lineNo, line in enumerate(ratings):
    items = split(' |,|\t', line.strip())
    userId = items[0]
    itemId = items[1]
    if itemId not in item_times:
        item_times[itemId] = 1
    else:
        item_times[itemId] += 1
    
    rating  = items[2]
    trainingData[userId][itemId]=float(rating)

for i,user in enumerate(trainingData):
    for item in trainingData[user]:
        trainingSet_i[item][user] = trainingData[user][item]

item_keys = []
item_value = []
trainingData_itemavg = defaultdict(dict)
for item in trainingSet_i:
    trainingData_itemavg[item] = sum(trainingSet_i[item].values())/len(trainingSet_i[item])
    item_keys.append(item)
    item_value.append(sum(trainingSet_i[item].values())/len(trainingSet_i[item]))

item_targets = []
for i in range(len(item_keys)):
    if item_value[i] < threshold_rating and int(item_times[str(item_keys[i])]) > threshold_frequency:
        item_targets.append(item_keys[i])

targets = random.sample(list(item_targets),targets_number)


#item_targets = np.where(np.array(list(trainingData_itemavg.values()))<threshold)

print(targets)

with open('epinions_targets.txt', 'w') as f:
    for i in range(len(targets)):
        f.write(str(targets[i])+'\n')
