# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 17:07:01 2020

@author: lidazhang
"""

import numpy as np
import os
import random
import pickle
import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell, GRUCell, static_rnn
#from tensorflow.compat.v1.nn.rnn_cell import BasicLSTMCell, GRUCell
from tensorflow.nn import dynamic_rnn
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from flip_gradient import flip_gradient


def correlation_coefficient(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = np.mean(x)
    my = np.mean(y)
    xm, ym = x-mx, y-my
    r_num = np.sum(np.multiply(xm,ym))
    r_den = np.sqrt(np.multiply(np.sum(np.square(xm)), np.sum(np.square(ym))))
    r = r_num / r_den

    r = np.maximum(np.minimum(r, 1.0), -1.0)
    return r

def pad_data(data, scaler, series_len, num_features):
    data_padded = np.zeros((0,series_len,num_features))
    for i in range(len(data)):
            l = data[i].shape[0]
            data_padded = np.concatenate((data_padded,np.pad(scaler.transform(data[i]), ((series_len-l,0),(0,0)), 'constant',
                                                             constant_values = -3 ).reshape(1,series_len,num_features)), axis=0)
    return data_padded


class RNN:
     def __init__(self, is_training = True):
        self.max_grad_norm = 5

        self.learning_rate = 0.003
        self.unit_lstm = 30
        self.unit = 30
        self.drop_rate = 0.7
        self.batch_size = 191

        #1:383 3:245 6:101 7:74 9:62 10:44 11:138 12:67 13:87 14:108 15:166
        #1:191 3:122 6:202 7:148 9:124 10:88 11:138 12:67 13:87 14:108 15:166

        self.length = 100
        self.fea_dim = 9
        self.raw = 4
        self.num_sub = 3

        self.input = tf.placeholder(tf.float32, [None, self.length, self.fea_dim])
        self.sbp_label = tf.placeholder(tf.float32, [None,1])
        self.dbp_label = tf.placeholder(tf.float32, [None,1])
        self.domain = tf.placeholder(tf.int32, [None,self.num_sub])
        self.l = tf.placeholder(tf.float32, [])
        self.train = tf.placeholder(tf.bool, [])

        with tf.variable_scope('feature_extractor'):
            inputs = tf.transpose(self.input,[1,0,2])
            inner_cell = BasicLSTMCell(self.unit_lstm)
            outputs, final_state = dynamic_rnn(inner_cell, inputs, time_major=True, dtype = tf.float32)

            if is_training:
                keep_prob = tf.constant(self.drop_rate)
            else:
                keep_prob = tf.constant(1.0)
            outputs = tf.nn.dropout(outputs, keep_prob) #keep_prob = tf.constant(1.0)

            #idx = tf.range(self.batch_size)*tf.shape(outputs)[1] + (self.seq_len - 1)
            #output = tf.gather(tf.reshape(outputs, [-1, self.unit]), idx)
            outputs = tf.transpose(outputs, [1,0,2]) #batch, seq, hidden
            output = tf.slice(outputs, [0, self.length-1, 0], [-1, 1, self.unit_lstm])
            output = tf.squeeze(output, axis=1)

            output = tf.layers.dense(output, self.unit, activation=tf.nn.relu, name='shared_dense')

        with tf.variable_scope('label_predictor'):
            sbp_dense1 = tf.layers.dense(output, self.unit, activation=tf.nn.relu, name='sbp_dense1')
            dbp_dense1 = tf.layers.dense(output, self.unit, activation=tf.nn.relu, name='dbp_dense1')

            sbp_dense = tf.layers.dense(sbp_dense1, self.unit, activation=tf.nn.relu, name='sbp_dense2')
            dbp_dense = tf.layers.dense(dbp_dense1, self.unit, activation=tf.nn.relu, name='dbp_dense2')

            sbp_dense = tf.layers.dense(sbp_dense, self.unit, activation=tf.nn.relu, name='sbp_dense3')
            dbp_dense = tf.layers.dense(dbp_dense, self.unit, activation=tf.nn.relu, name='dbp_dense3')

            #sbp_dense4 = tf.layers.dense(sbp_dense3, 70, activation=tf.nn.relu,  name='sbp_dense4')
            #dbp_dense4 = tf.layers.dense(dbp_dense3, 70, activation=tf.nn.relu,  name='dbp_dense4')

            #sbp_dense = tf.layers.dense(sbp_dense4, 70, activation=tf.nn.relu,  name='sbp_dense5')
            #dbp_dense = tf.layers.dense(dbp_dense4, 70, activation=tf.nn.relu,  name='dbp_dense5')

            self.sbp = tf.layers.dense(sbp_dense, 1, name='sbp_out')
            self.dbp = tf.layers.dense(dbp_dense, 1, name='dbp_out')
            loss1 = tf.losses.mean_squared_error(self.sbp_label, self.sbp)
            loss2 = tf.losses.mean_squared_error(self.dbp_label, self.dbp)
            self.pred_losses = loss1+loss2

        with tf.variable_scope('domain_predictor'):
            feat = flip_gradient(output, self.l)
            dom = tf.layers.dense(feat, 30, activation=tf.nn.relu, name='domain1')
            dom = tf.layers.dense(dom, self.num_sub, name='domain2')

            self.domain_pred = tf.nn.softmax(dom)
            self.domain_losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.domain_pred, labels=self.domain)

        # learning_rate = tf.placeholder(tf.float32, [])
        # pred_loss = tf.reduce_mean(self.pred_loss)
        # domain_loss = tf.reduce_mean(self.domain_loss)
        # total_loss = pred_loss + domain_loss
        #
        # regular_train_op = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(pred_loss)
        # dann_train_op = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(total_loss)
        # correct_domain_pred = tf.equal(tf.argmax(model.domain, 1), tf.argmax(model.domain_pred, 1))
        # domain_acc = tf.reduce_mean(tf.cast(correct_domain_pred, tf.float32))


class Data_generator:
    def __init__(self, minute):
        data_dir={}
        label_dir={}
        self.source_idx = source_idx = 11
        self.target_idx = target_idx = 13
        self.test_idx = test_idx = 7
        subject_list = [source_idx, target_idx, test_idx]
        train_list = [source_idx, target_idx]

        order_file = '/home/grads/l/lidazhang/bioz/Subject'+str(source_idx)+'/order'
        if os.path.exists(order_file):
            with open(order_file, 'rb') as order:
                idx = pickle.load(order)
                self.source_order = idx
            #print('Load order:',idx)
        order_file = '/home/grads/l/lidazhang/bioz/Subject'+str(target_idx)+'/order'
        if os.path.exists(order_file):
            with open(order_file, 'rb') as order:
                idx = pickle.load(order)
                self.target_order = idx
            #print('Load order:',idx)

        order_file = '/home/grads/l/lidazhang/bioz/Subject'+str(test_idx)+'/order'
        if os.path.exists(order_file):
            with open(order_file, 'rb') as order:
                idx = pickle.load(order)
                self.test_order = idx
            #print('Load order:',idx)

        series_len = 100
        num_features = 9

        for s in subject_list:
            subject = 'Subject'+str(s)

            dir_path = '/home/grads/l/lidazhang/bioz/'+subject
            f = open(dir_path+'/labels_100.pckl', 'rb')
            labels = pickle.load(f, encoding='latin1')
            label_dir[s]=np.array(labels)
            f.close()

            f = open(dir_path+'/list_data_time_100.pckl', 'rb')
            list_data = pickle.load(f, encoding='latin1')
            f.close()

            for i in range(len(list_data)):
                for j in range(4):
                    list_data[i] = np.concatenate((list_data[i],np.reshape(np.gradient(list_data[i][:,j]),
                                                                          (list_data[i].shape[0], 1))), axis=1)
            data_dir[s] = list_data

        count = []
        for b in data_dir[test_idx]:
            count.append(len(b))
        avg_len = np.sum(count)/len(count)/100
        t_beats = int(minute*60/avg_len)+10
        print('source beats:', len(self.source_order), len(self.target_order))
        print('test beats:', t_beats, 'total:', len(count), 'avg len:', avg_len)

        batch_list = [45,50,55]#[45,50,55] [55,60,65] [90,100,110]
        batch_mod = [t_beats%i for i in batch_list]
        self.test_batch = batch_list[np.argmin(batch_mod)]
        self.batch_num = t_beats//self.test_batch
        self.source_batch = 100
        self.target_batch = 100#len(self.source_order)//(self.batch_num)
        print('test retrain beats:', self.test_batch*self.batch_num , 'test batch:', self.test_batch)
        print('batch num:',self.batch_num, 'sourch batch:',self.source_batch, 'target batch:', self.target_batch)

        label_all = []
        for ss in label_dir.keys():
          label_all.extend(label_dir[ss])
        self.label_scaler = MinMaxScaler(feature_range=(0, 1))
        label_all = np.array(label_all)
        self.label_scaler.fit(label_all)

        train_data=data_dir[self.source_idx]

        scale_data = []
        for beat in train_data:
             scale_data.extend(beat)
        self.standard_data_scaler = StandardScaler()
        self.standard_data_scaler.fit(scale_data)

        for ss in label_dir.keys():
             data_dir[ss] = pad_data(data_dir[ss], self.standard_data_scaler, series_len, num_features)
             label_dir[ss] = self.label_scaler.transform(label_dir[ss])
        self.data = data_dir
        self.label = label_dir


    def generate(self, ite, da_train=True):
        x0, y0, x1, y1, x2, y2 = [], [], [], [], [], []
        s_batch = self.source_batch
        t_batch = self.target_batch
        test_batch = self.test_batch
        if da_train:
            x0=self.data[self.source_idx][self.source_order[s_batch*ite:s_batch*(ite+1)]]
            y0=self.label[self.source_idx][self.source_order[s_batch*ite:s_batch*(ite+1)]]

            x1=self.data[self.target_idx][self.target_order[t_batch*ite:t_batch*(ite+1)]]
            y1=self.label[self.target_idx][self.target_order[t_batch*ite:t_batch*(ite+1)]]

        if test_batch*(ite+1)<len(self.test_order):
            x2=self.data[self.test_idx][self.test_order[test_batch*ite:test_batch*(ite+1)]]
            y2=self.label[self.test_idx][self.test_order[test_batch*ite:test_batch*(ite+1)]]
        return np.array(x0), np.array(y0), np.array(x1), np.array(y1), np.array(x2), np.array(y2)


if __name__== "__main__":
    random.seed(1)
    tf.set_random_seed(1)
    np.random.seed(1)

    series_len = 100
    num_features = 9
    #batch_size = 74

    data_generator = Data_generator(4)

    # Build the model graph
    #graph = tf.get_default_graph()
    with tf.Graph().as_default(), tf.Session() as sess:
        model = RNN()

        learning_rate = tf.placeholder(tf.float32, [])
        pred_loss = tf.reduce_mean(model.pred_losses)
        domain_loss = tf.reduce_mean(model.domain_losses)
        total_loss = pred_loss + domain_loss

        #regular_train_op = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(pred_loss)
        regular_train_op = tf.train.AdamOptimizer(0.0015).minimize(pred_loss)
        dann_train_op = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(total_loss)
        correct_domain_pred = tf.equal(tf.argmax(model.domain, 1), tf.argmax(model.domain_pred, 1))
        domain_acc = tf.reduce_mean(tf.cast(correct_domain_pred, tf.float32))

        domain_labels = np.vstack([np.tile([1., 0., 0.], [data_generator.source_batch, 1]),
                                   np.tile([0., 1., 0.], [data_generator.target_batch, 1]),
                                   np.tile([0., 0., 1.], [data_generator.test_batch, 1])])

        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver()
        save_path = 'checkpoints1_4min/dann_'+str(data_generator.test_idx)
        da_batch_num = data_generator.batch_num
        num_steps = 400
        for i in range(200):
            print(i)
            p = float(i) / num_steps
            l = 2. / (1. + np.exp(-10. * p)) - 1
            lr = 0.01 / (1. + 10 * p)**0.75

            for b in range(da_batch_num):
                x0, y0, x1, y1, x2, y2 = data_generator.generate(b)
                #print(len(x0),len(x1),len(x2))
                if len(x0)==0:
                    train_batch = b
                    break
                #print(1111,len(domain_labels),len(x0), len(x1), len(x2))
                X = np.vstack([x0, x1, x2])
                y = np.vstack([y0, y1, y2])

                _, batch_loss, dom_loss, pr_loss, dom_acc = sess.run(
                    [dann_train_op, total_loss, domain_loss, pred_loss, domain_acc],
                    feed_dict={model.input: X, model.dbp_label: np.expand_dims(y[:,0],-1), model.sbp_label: np.expand_dims(y[:,1],-1),
                               model.domain: domain_labels, model.train: True, model.l: l, learning_rate: lr})

            print(batch_loss, dom_loss, pr_loss, dom_acc)

        saver.save(sess, save_path)
        saver.restore(sess, save_path)
