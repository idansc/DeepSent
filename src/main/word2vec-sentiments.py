# gensim modules

from gensim.models import Doc2Vec

from parserdata import Parser
from parserdata import Batcher
from parserdata import TaggedLineSentence

import numpy as np

from sklearn.linear_model import LogisticRegression

import tensorflow as tf

import logging
import sys


config ={
    'log_channel':sys.stdout,
    'dov2vec_sources': {'../resource/test-neg.txt':'TEST_NEG', '../resource/test-pos.txt':'TEST_POS',
           '../resource/train-neg.txt':'TRAIN_NEG', '../resource/train-pos.txt':'TRAIN_POS',
           '../resource/train-unsup.txt':'TRAIN_UNS'},
    'load_doc2vec': True,
    'classifier_class':False,
    'n_samples': 25000,
    'doc2vec_dim': 100,
    'batch_size': 100
}

#Logging init
log = logging.getLogger()
log.setLevel(logging.INFO)

ch = logging.StreamHandler(config['log_channel'])
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
log.addHandler(ch)


log.info('source load')
sources = config['dov2vec_sources']

parser = Parser()

log.info('TaggedDocument')
sentences = TaggedLineSentence(sources)

log.info('D2V')
model = Doc2Vec(min_count=1, window=10, size=config['doc2vec_dim'], sample=1e-4, negative=5, workers=7)
model.build_vocab(sentences.to_array())

if config['load_doc2vec']:
    model = Doc2Vec.load('../resource/imdb.d2v')
else:
    log.info('Epoch')
    for epoch in range(10):
        log.info('EPOCH: {}'.format(epoch))
        model.train(sentences.sentences_perm())

    log.info('Model Save')
    model.save('./imdb.d2v')

log.info('Sentiment')

train_arrays = np.zeros((25000, config['doc2vec_dim']))

train_labels = np.zeros(25000)
train_labels_binary =  np.zeros(25000)

test_arrays = np.zeros((25000, config['doc2vec_dim']))
test_labels_binary =  np.zeros(25000)


train_labels1 = np.zeros(25000)
train_labels2 = np.zeros(25000)
train_labels3 = np.zeros(25000)
train_labels4 = np.zeros(25000)
train_labels7 = np.zeros(25000)
train_labels8 = np.zeros(25000)
train_labels9 = np.zeros(25000)
train_labels10 = np.zeros(25000)


test_labels1 = np.zeros(25000)
test_labels2 = np.zeros(25000)
test_labels3 = np.zeros(25000)
test_labels4 = np.zeros(25000)
test_labels7 = np.zeros(25000)
test_labels8 = np.zeros(25000)
test_labels9 = np.zeros(25000)
test_labels10 = np.zeros(25000)



for i in range(12500):
    #Train
    prefix_train_pos = 'TRAIN_POS_' + str(i)
    prefix_train_neg = 'TRAIN_NEG_' + str(i)
    train_arrays[i] = model.docvecs[prefix_train_pos]
    train_arrays[12500 + i] = model.docvecs[prefix_train_neg]

    train_labels = parser.train_pos_scores[i]
    train_labels[12500 + i] = parser.train_neg_scores[i]

    train_labels_binary[i] = 1
    train_labels_binary[12500 + i] = 0
    if config['classifier_class']:
        train_labels10[i] = 1 if parser.train_pos_scores[i] == 10 else 0
        train_labels10[12500 + i] = 0

        train_labels9[i] = 1 if parser.train_pos_scores[i] == 9 else 0
        train_labels9[12500 + i] = 0

        train_labels8[i] = 1 if parser.train_pos_scores[i] == 8 else 0
        train_labels8[12500 + i] = 0

        train_labels7[i] = 1 if parser.train_pos_scores[i] == 7 else 0
        train_labels7[12500 + i] = 0

        train_labels1[i] = 0
        train_labels1[12500 + i] = 1 if parser.train_neg_scores[i] == 1 else 0

        train_labels2[i] = 0
        train_labels2[12500 + i] = 1 if parser.train_neg_scores[i] == 2 else 0

        train_labels3[i] = 0
        train_labels3[12500 + i] = 1 if parser.train_neg_scores[i] == 3 else 0

        train_labels4[i] = 0
        train_labels4[12500 + i] = 1 if parser.train_neg_scores[i] == 4 else 0

    #Test
    prefix_test_pos = 'TEST_POS_' + str(i)
    prefix_test_neg = 'TEST_NEG_' + str(i)
    test_arrays[i] = model.docvecs[prefix_test_pos]
    test_arrays[12500 + i] = model.docvecs[prefix_test_neg]

    test_labels = parser.test_pos_scores[i]
    test_labels[12500 + i] = parser.test_neg_scores[i]

    test_labels_binary[i] = 1
    test_labels_binary[12500 + i] = 0

    if config['classifier_class']:
        test_labels10[i] = 1 if parser.test_pos_scores[i] == 10 else 0
        test_labels10[12500 + i] = 0

        test_labels9[i] = 1 if parser.test_pos_scores[i] == 9 else 0
        test_labels9[12500 + i] = 0

        test_labels8[i] = 1 if parser.test_pos_scores[i] == 8 else 0
        test_labels8[12500 + i] = 0

        test_labels7[i] = 1 if parser.test_pos_scores[i] == 7 else 0
        test_labels7[12500 + i] = 0

        test_labels1[i] = 0
        test_labels1[12500 + i] = 1 if parser.test_neg_scores[i] == 1 else 0

        test_labels2[i] = 0
        test_labels2[12500 + i] = 1 if parser.train_neg_scores[i] == 2 else 0

        test_labels3[i] = 0
        test_labels3[12500 + i] = 1 if parser.train_neg_scores[i] == 3 else 0

        test_labels4[i] = 0
        test_labels4[12500 + i] = 1 if parser.train_neg_scores[i] == 4 else 0



classifier1 = LogisticRegression()
classifier2 = LogisticRegression()
classifier3 = LogisticRegression()
classifier4 = LogisticRegression()
classifier7 = LogisticRegression()
classifier8 = LogisticRegression()
classifier9 = LogisticRegression()
classifier10 = LogisticRegression()


log.info('Logistic Classifier')

classifier = LogisticRegression()
classifier_binary = LogisticRegression()

classifier_binary.fit(train_arrays, train_labels_binary)
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001)
print classifier_binary.score(test_arrays,test_labels_binary)

log.info('CNN')

##init CONv layer
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

##init max pool layer
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

test_arrays = np.reshape(test_arrays, (config['n_samples'],config['doc2vec_dim']))
train_arrays = np.reshape(test_arrays, (config['n_samples'],config['doc2vec_dim']))

train_labels = np.array([x-2 if x>4 else x for x in train_labels])
test_labels = np.array([x-2 if x>4 else x for x in test_labels])

# Define placeholders for input
x = tf.placeholder(tf.float32, shape=(None, config['doc2vec_dim']))
y_ = tf.placeholder(tf.float32, shape=(None, config['doc2vec_dim']))

with tf.variable_scope("linear1"):
    W = tf.get_variable("weights", (100,8), initializer=tf.random_normal_initializer(stddev=0.1))
    b = tf.get_variable("bias", [8], initializer=tf.constant_initializer(value=0.1, dtype=tf.float32))

    y = tf.nn.softmax(tf.matmul(x,W)+b)

with tf.Session() as sess:
    test_labels = tf.one_hot(test_labels, 8, on_value=1.0, off_value=0.0)
    test_labels = test_labels.eval()
    print test_labels
    train_labels = tf.one_hot(train_labels, 8, on_value=1.0, off_value=0.0)
    train_labels.eval()
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    for _ in range(1000):
        indices = np.random.choice(config['n_samples'],config['batch_size'])
        X_batch,y_batch = train_arrays[indices],train_labels[indices]
        sess.run(train_step, feed_dict={x: X_batch, y_: y_batch})
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(accuracy.eval(feed_dict={x: test_arrays, y_: test_labels}))


    #x_doc = tf.reshape(x, [-1,10,10,1])

    #h_conv1 = tf.nn.relu(tf.nn.conv2d(x_doc, W_conv1) + b_conv1)



'''
x_image = tf.reshape(x, [-1,10,10,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([1, 1, 16, 8])
b_conv2 = bias_variable([8])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_avg_pool = tf.nn.avg_pool(x, ksize=[1, 5, 5, 1],
                        strides=[1, 1, 1, 1], padding='SAME')

y_conv=tf.nn.softmax(h_avg_pool)

with tf.Session() as sess:
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    sess.run(tf.initialize_all_variables())
    data = Batcher(x, y_, 50)
    for i in range(20000):
        batch = data.next_batch()
        if i%100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x:batch[0], y_: batch[1]})
            print("step %d, training accuracy %g"%(i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1]})

    #results
    print("test accuracy %g"%accuracy.eval(feed_dict={
        x: test_arrays, y_: test_labels}))




classifier1.fit(train_arrays, train_labels1)
y1 = classifier1.predict_proba(test_arrays)
print y1[100][0],y1[100][1]
'''
'''
classifier2.fit(train_arrays, train_labels2)
classifier3.fit(train_arrays, train_labels3)
classifier4.fit(train_arrays, train_labels4)
classifier5.fit(train_arrays, train_labels5)
classifier6.fit(train_arrays, train_labels6)
classifier7.fit(train_arrays, train_labels7)
classifier8.fit(train_arrays, train_labels8)
'''


'''
print classifier1.score(test_arrays, test_labels1)
print classifier2.score(test_arrays, test_labels2)
print classifier3.score(test_arrays, test_labels3)
print classifier4.score(test_arrays, test_labels4)

for i in range(9):

y = classifier.predict(test_arrays)


y1 = classifier1.predict(test_arrays)
print "classify 1" , accuracy_score(y1,test_labels1)

y2 = classifier2.predict(test_arrays)
print "classify 2" , accuracy_score(y2,test_labels2)

y3 = classifier3.predict(test_arrays)
print "classify 3" , accuracy_score(y3,test_labels3)

y4 = classifier4.predict(test_arrays)
print "classify 4" , accuracy_score(y4,test_labels4)

y5 = classifier5.predict(test_arrays)
print "classify 5" , accuracy_score(y5,test_labels5)

y6 = classifier6.predict(test_arrays)
print "classify 6" , accuracy_score(y6,test_labels6)

y7 = classifier7.predict(test_arrays)
print "classify 7" , accuracy_score(y7,test_labels7)

y8 = classifier8.predict(test_arrays)
print "classify 8" , accuracy_score(y8,test_labels8)
'''

'''
y1 = classifier1.predict_proba(test_arrays)
#print "classify 1" , accuracy_score(y1,test_labels1)

y2 = classifier2.predict_proba(test_arrays)
#print "classify 2" , accuracy_score(y2,test_labels2)

y3 = classifier3.predict_proba(test_arrays)
#print "classify 3" , accuracy_score(y3,test_labels3)

y4 = classifier4.predict_proba(test_arrays)
#print "classify 4" , accuracy_score(y4,test_labels4)

y5 = classifier5.predict_proba(test_arrays)
#print "classify 5" , accuracy_score(y5,test_labels5)

y6 = classifier6.predict_proba(test_arrays)
#print "classify 6" , accuracy_score(y6,test_labels6)

y7 = classifier7.predict_proba(test_arrays)
print y7
#print "classify 7" , accuracy_score(y7,test_labels7)

y8 = classifier8.predict_proba(test_arrays)
#print "classify 8" , accuracy_score(y8,test_labels8)


yfinal = numpy.zeros((25000,8))

for i in range(25000):
    yfinal[i] = [y1[i][1],y2[i][1],y3[i][1],y4[i][1],y5[i][1],y6[i][1],y7[i][1],y8[i][1]]

#finalClassifier.fit(yfinal,test_labels)

for i in range(25000):
    if i<12500:
        print yfinal[i][1], " GT:", parser.train_pos_scores[i]
    if i>12500:
        print yfinal[i][1] , " GT:",parser.train_neg_scores[i-12500]

for i in range(25000):
    results[i] = 1 if yfinal[i].argmax() < 4 else 0
print accuracy_score(results,test_labels)

for i in range(25000):
    count_pos = 0
    count_neg = 0
    if y1[i][1]>0.5:
        count_pos += 1
    if y2[i][1]>0.5:
        count_pos += 1
    if y3[i][1]>0.5:
        count_pos += 1
    if y4[i][1]>0.5:
        count_pos += 1
    if y5[i][1]>0.5:
        count_neg += 1
    if y6[i][1]>0.5:
        count_neg += 1
    if y7[i][1]>0.5:
        count_neg += 1
    if y8[i][1]>0.5:
        count_neg += 1
    results[i] = 1 if count_pos > count_neg else 0
print accuracy_score(results,test_labels)


for i in range(25000):
    results[i] = 1 if  y2[i] == 1 or y3[i]==1 or y4[i]==1  else 0
print accuracy_score(results,test_labels)

for i in range(25000):
    results[i] = 1 if y1[i]+y2[i]+y3[i]+y4[i]>y5[i]+y6[i]+y7[i]+y8[i] else 0
print accuracy_score(results,test_labels)
'''
#print finalClassifier.score(yfinal,test_labels)



print
#results = [1 if x>i else 0 for x in classifier.predict(test_arrays)]

