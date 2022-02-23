# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 21:33:14 2018

@author: WZW
"""
import tensorflow as tf
import numpy as np
import os
from collections import defaultdict
from re import compile,findall,split

from recommendation.QRec import QRec
from recommendation.util.config import ModelConf

import random

tf.compat.v1.disable_eager_execution()

attacksize = 0.1
sess = tf.compat.v1.InteractiveSession()

mb_size = 20
Z_dim = 100

targets = []
with open('targets.txt') as f:
    content = f.readlines()
    for item in content:
        item = item.strip('\n')
        targets.append(item)




with open("dataset/Movielens/trainset.txt") as f:
    ratings = f.readlines()

order = ('0 1 2').strip().split()
trainingData = defaultdict(dict)
trainingSet_i = defaultdict(dict)
user_num = 0
item_num = 0
item_most = 0
for lineNo, line in enumerate(ratings):
    items = split(' |,|\t', line.strip())
    userId = items[int(order[0])]
    itemId = items[int(order[1])]
    if int(itemId) > item_most:
        item_most = int(itemId)
    rating  = items[int(order[2])]
    trainingData[userId][itemId]=float(rating)/float(5)

for i,user in enumerate(trainingData):
    for item in trainingData[user]:
        trainingSet_i[item][user] = trainingData[user][item]
        
#print(trainingSet_i)
print(len(trainingData),item_most)
x = np.zeros((len(trainingData),item_most))
y = np.zeros((len(trainingData),2))

for user in trainingData:
    for item in trainingData[user]:
        x[int(user)-1][int(item)-1] = trainingData[user][item]
        
filename = 'attackDataset/GSA_GANs_profiles.txt'
with open(filename,'w') as f:
    f.writelines(ratings)

#get_variabel() 获取已存在的变量（要求不仅名字，而且初始化方法等各个参数都一样），如果不存在，就新建一个。 
#tf.contrib.layers.xavier_initializer() 初始化权重矩阵

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random.normal(shape=size, stddev=xavier_stddev)

def weight_var(shape, name):
    return tf.compat.v1.get_variable(name=name, shape=shape,   initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"))

#tf.constant_initializer() 初始化一个常数
def bias_var(shape, name):
    return tf.compat.v1.get_variable(name=name, shape=shape, initializer=tf.compat.v1.constant_initializer(0))


# discriminater net（判别模型）
X = tf.compat.v1.placeholder(tf.float32, shape=[None, item_most], name='X')

D_W1 = tf.Variable(xavier_init([item_most, 128]))
D_b1 = tf.Variable(tf.zeros(shape=[128]))

D_W2 = tf.Variable(xavier_init([128, 1]))
D_b2 = tf.Variable(tf.zeros(shape=[1]))

theta_D = [D_W1, D_W2, D_b1, D_b2]


Z = tf.compat.v1.placeholder(tf.float32, shape=[None, Z_dim])

G_W1 = tf.Variable(xavier_init([Z_dim, 128]))
G_b1 = tf.Variable(tf.zeros(shape=[128]))

G_W2 = tf.Variable(xavier_init([128, item_most]))
G_b2 = tf.Variable(tf.zeros(shape=[item_most]))

theta_G = [G_W1, G_W2, G_b1, G_b2]

def generator(z):
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)
    return G_prob


def discriminator(x):
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    out = tf.matmul(D_h1, D_W2) + D_b2
    return out

def cal_utility_rating(targets):
    res = []
    with open('./results/simulator_prediction.txt') as f:
        content = f.readlines()
        temp = []
        for rate in content:
            rate = rate.strip('\n')
            rate = rate.split(' ')
            if rate[1] in targets:
                res.append(float(rate[3]))
    
    print(res)
    print(len(res))
    result = (5 - np.array(res)).sum() / len(res)
    print('cal_utility_rating', result)
    return result

def cal_utility_ranking(targets):
    with open('./results/simulator_prediction.txt') as f:
        content = f.readlines()
        temp = []
        for rate in content[1:]:
            rate = rate.strip('\n')
            rate = rate.split(' ')
            row = []
            row.append(rate[0])
            row.append(rate[1])
            row.append(rate[2])
            row.append(rate[3])
            temp.append(row)
        user_item = {}
        for user,item,rate,prediction in temp:
            if user in user_item:
                if item in targets:
                    user_item[user][0].append(float(prediction))
                else:
                    user_item[user][1].append(float(prediction))
            else:
                user_item[user] = [[],[]]
                if item in targets:
                    user_item[user][0].append(float(prediction))
                else:
                    user_item[user][1].append(float(prediction))
        temp_result = 0
        length = 0
        for key in user_item:
            prediction_targets = user_item[key][0]
            prediction_nontargets = user_item[key][1]

            for targets_score in prediction_targets:
                temp_result += (targets_score - np.array(prediction_nontargets)).sum()
                length += len(prediction_nontargets)
        result = temp_result / length

        print(result)
        return result

def run_recommendation(algorithm):
    conf= ModelConf('./recommendation/config/' + algorithm + '.conf')
    recSys = QRec(conf)
    recSys.execute()



G_sample = generator(Z)
D_real = discriminator(X)
D_fake = discriminator(G_sample)

utility_attack = cal_utility_ranking(targets)

D_loss = tf.reduce_mean(input_tensor=D_real) - tf.reduce_mean(input_tensor=D_fake)
G_loss = -tf.reduce_mean(input_tensor=D_fake) - utility_attack

D_solver = (tf.compat.v1.train.RMSPropOptimizer(learning_rate=1e-4)
            .minimize(-D_loss, var_list=theta_D))
G_solver = (tf.compat.v1.train.RMSPropOptimizer(learning_rate=1e-4)
            .minimize(G_loss, var_list=theta_G))

clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in theta_D]
#tf.nn.relu()激活函数

def sample_Z(m, n):
    '''Uniform prior for G(Z)'''
    return np.random.uniform(-1., 1., size=[m, n])
    
sess.run(tf.compat.v1.global_variables_initializer())

user_num = len(trainingData)
has_user = 0
i = 0
for it in range(1000000):
    print(it)
    if it % 2000 == 0:
        run_recommendation('BasicMF')
        utility_attack = cal_utility_ranking(targets)
    if it % 1000 == 0:
        # G_sample = generator(Z)
        
        samples = sess.run(G_sample, feed_dict={
                        Z: sample_Z(int(len(trainingData)), Z_dim)})
        print(samples)

    
    
    choose_user = random.sample(range(1,len(trainingData)), 100)

    batch_xs = []
    for i in range(32):
        batch_xs.append(x[choose_user[i]])
    #print(batch_xs)
    _, D_loss_curr, _ = sess.run(
        [D_solver, D_loss, clip_D],
        feed_dict={X: np.array(batch_xs), Z: sample_Z(mb_size, Z_dim)}
    )

    _, G_loss_curr = sess.run(
    [G_solver, G_loss],
    feed_dict={Z: sample_Z(mb_size, Z_dim)}
    )
    # if abs(G_loss_curr) < 1e-3 or abs(D_loss_curr) < 1e-3 or (it % 100 == 0 and it % 100 != 0):
    if it % 100 == 0 and it / 100 != 0:
        attackitem = random.sample(targets,1)
        if not os.path.exists('attackDataset/'):
            os.makedirs('attackDataset/')
        print(it)
        print(G_loss_curr,D_loss_curr)
        print(samples)

        true_num = user_num
        with open('attackDataset/GSA_GANs_profiles.txt','a') as f:
            rating_gan = []
            for i in samples:
                if np.sum(i > 0.2) > 10:
                    has_user += 1
                    user_num += 1
                    print(has_user)
                    for m,n in enumerate(list(i)):
                        if m+1 == attackitem[0]:
                            rating_gan.append(str(user_num)+' '+str(m+1)+' '+str(round(5))+'\n')
                        elif n >= 0.2:
                            #ratings.append(str(user_num)+' '+str(m+1)+' '+str(round(n*5))+'\n')
                            rating_gan.append(str(user_num)+' '+str(m+1)+' '+str(round(n*5))+'\n')
                        else:
                            continue
                #ratings.append(str(user_num)+' '+str(305)+' '+str(5)+'\n')
            
            f.writelines(rating_gan)
        if has_user == attacksize *int(len(trainingData)):
            break
        with open('attackDataset/GSA_GANs_labels.txt','a') as f:
            for user in range(user_num):
                if user <= true_num:
                    f.write(str(user+1)+' '+ '0' +'\n')
                else:
                    f.write(str(user+1)+' '+ '1' +'\n')


    if it % 1000 == 0:
        print('Iter: {}'.format(it))
        print('D loss: {:.4}'.format(D_loss_curr))
        print('G_loss: {:.4}'.format(G_loss_curr))
        print()
    
