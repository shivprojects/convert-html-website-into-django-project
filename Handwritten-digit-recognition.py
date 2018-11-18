# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 02:01:12 2018

@author: shivr
"""

import os
import numpy as np
import pandas as pd
from skimage.io import imread
#from sklearn.metrics import accuracy_score
import tensorflow as tf
import pylab

root_dir = '<root dir where data directory present>'

print ("Running ...")
# To clear the defined variables and operations of the previous cell
tf.reset_default_graph() 

# 1. load images data 
# 2. split the train and test set
# 3. defined variables for neural network
# 4. Training started
# 5. accuracy on trained models with test set
# 6. Testing with known images
 
def one_hot_encoding(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors"""
    
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1    
    return labels_one_hot

def batch_creator(batch_size):
    """Create batch with random samples and return appropriate format"""
    dataset_length = len(train_x) # 32300
    batch_mask = rng.choice(dataset_length, batch_size)  # created 128 numbers between 1 - 32300
    batch_x = train_x[batch_mask].reshape(-1, input_num_units)

    """Convert values to range 0-1"""
    batch_x = batch_x / batch_x.max()
    
    batch_y = train.loc[batch_mask, 'label'].values
    batch_y = one_hot_encoding(batch_y)
    return batch_x, batch_y


def pixels_to_numbers(dataset_name):
    temp = []
    for img_name in eval(dataset_name).filename:
        image_path = os.path.join(data_dir, 'Images', dataset_name, img_name)
        img = imread(image_path, flatten=True)
        img = img.astype('float32')
        temp.append(img)
    return np.stack(temp)



def neural_network_model(data):
    
    
    #weight and biases
    #
    n_nodes_hl1 = 1000
    n_nodes_hl2 = 700
    n_nodes_hl3 = 500
    n_output_l = 10 # output
    # (input_data * weight) + biases
    hidden_1_layer = {'weights' : tf.Variable(tf.random_normal([input_num_units,n_nodes_hl1],seed=seed,name="hidden_1_layer-weight")), 
                      'biases' : tf.Variable(tf.random_normal([n_nodes_hl1],seed=seed,name="hidden_1_layer-bias"))}
#    hidden_2_layer = {'weights' : tf.Variable(tf.random_normal([n_nodes_hl1,n_nodes_hl2],seed=seed)), 
#                      'biases' : tf.Variable(tf.random_normal([n_nodes_hl2],seed=seed))}
#    hidden_3_layer = {'weights' : tf.Variable(tf.random_normal([n_nodes_hl2,n_nodes_hl3],seed=seed)), 
#                      'biases' : tf.Variable(tf.random_normal([n_nodes_hl3],seed=seed))}
    output_layer = {'weights' : tf.Variable(tf.random_normal([n_nodes_hl1,n_output_l],seed=seed,name="output-weight")), 
                      'biases' : tf.Variable(tf.random_normal([n_output_l],seed=seed,name="output-biases"))}

    # l1 =  (input_data * weight) + biases
    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']),hidden_1_layer['biases'],name="l1-mutmul")
    l1 = tf.nn.relu(l1,name="l1-relu")
    
    # l2 = (l1 * weight) + biases
#    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']),hidden_2_layer['biases'])
#    l2 = tf.nn.relu(l2)
    
    # l3 = (l2 * weight) + biases
#    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']),hidden_3_layer['biases'])
#    l3 = tf.nn.relu(l3)

    # output = (l3 * weight) + biases
    output = tf.add(tf.matmul(l1,output_layer['weights']),output_layer['biases'],name="output-add")
    
    return output


# To stop potential randomness 
seed = 128
rng = np.random.RandomState(seed)

data_dir = os.path.join(root_dir, 'data')

# Load images data --------------
train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    
print ("get_images_data ...")
dataset = pixels_to_numbers("train")

# split the train and test set --------------

from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(dataset, train.label.values ,test_size=0.3,shuffle=False)

# defined variables for neural network ------------------------

# number of neurons in each layer
input_num_units = 28*28
output_num_units = 10

# define placeholders
x = tf.placeholder(tf.float32, [None, input_num_units], name="x")
y = tf.placeholder(tf.float32, [None, output_num_units],name="y")

# set remaining variables
epochs = 5
batch_size = 128
learning_rate = 0.01

output_layer = neural_network_model(x)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer,labels=y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Training started -------------------------------------
print ("training started ...")
with tf.Session() as sess:
    init = tf.init = tf.global_variables_initializer()
    sess.run(init)
    total_batch = int(train.shape[0]/batch_size)
    for epoch in range(epochs):
        avg_cost = 0
        for i in range(total_batch):
            batch_x, batch_y = batch_creator(batch_size)
            _, c = sess.run([optimizer, cost], feed_dict = {x: batch_x, y: batch_y})
            
            avg_cost += c / total_batch
            
        print ("Epoch:", (epoch+1), "cost =", "{:.5f}".format(avg_cost))
    
    print ("\nTraining complete!")

    writer = tf.summary.FileWriter('./graphs', sess.graph)
    #tensorboard --logdir="./graphs"

    # accuracy on trained models with test set
    pred_temp = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(pred_temp, "float"))
    print ("Validation Accuracy:", accuracy.eval({x: test_x.reshape(-1, input_num_units), y: one_hot_encoding(test_y)}))
    predict = tf.argmax(output_layer, 1)

    # Testing with known images ------------------------------
    
    validate = pd.read_csv(os.path.join(data_dir, 'validate.csv'))
    validate_x = pixels_to_numbers("validate")
    predictions = predict.eval({x: validate_x.reshape(-1, input_num_units)})
    for img_name,pred in zip(validate.filename,predictions):
        filepath = os.path.join(data_dir, 'Images', 'validate', img_name)
        img = imread(filepath, flatten=True)
        print ("Prediction is: ",pred )
        pylab.imshow(img, cmap='gray')
        pylab.axis('off')
        pylab.show()
