# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 11:17:34 2019

@author: lizhenping
"""
'''
demo 例子说明
随机生成一个数组
[1,2,3,4,5]
输入GRU网络中
标记值为
[1,2,0,0,0]
规则为输入数据，经过GRU预测输出数据为输入数据的前两位
'''
import numpy as np
# Seed random
np.random.seed(0)
from random import randint
from numpy import array
from numpy import argmax
from itertools import chain 
 
# generate a sequence of random integers
def generate_sequence(length, n_unique):
	return [randint(0, n_unique-1) for _ in range(length)]
 
# one hot encode sequence
def one_hot_encode(sequence, n_unique):
	encoding = list()
	for value in sequence:
		vector = [0 for _ in range(n_unique)]
		vector[value] = 1
		encoding.append(vector)
	return array(encoding)
 
# decode a one hot encoded string
def one_hot_decode(encoded_seq):
	return [argmax(vector) for vector in encoded_seq]
 
# prepare data for the LSTM
def get_pair(n_in, n_out, n_unique):
	# generate random sequence
	sequence_in = generate_sequence(n_in, n_unique)
	sequence_out = sequence_in[:n_out] + [0 for _ in range(n_in-n_out)]
	# one hot encode

	return sequence_in,sequence_out
 
# test generate random sequence
x,y = get_pair(5, 2, 50)




# Activation functions
# NOTE: Derivatives are calculated using outcomes of their primitives (which are already calculated during forward prop).
def sigmoid(input, deriv=False):
    if deriv:
        return input*(1-input)
    else:
        return 1 / (1 + np.exp(-input))

def tanh(input, deriv=False):
    if deriv:
        return 1 - input ** 2
    else:
        return np.tanh(input)

# Derivative is directly calculated in backprop (in combination with cross-entropy loss function).
def softmax(input):
    # Subtraction of max value improves numerical stability.
    e_input = np.exp(input - np.max(input))
    return e_input / e_input.sum()
# definition the range of the input is 0-49
vocab_size = 50
# Hyper parameters
N, h_size, o_size = vocab_size, vocab_size, vocab_size # Hidden size is set to vocab_size, assuming that level of abstractness is approximately proportional to vocab_size (but can be set to any other value).
# Longer sequence lengths allow for lengthier latent dependencies to be trained.
seq_length = 25 

learning_rate = 1e-1

# Model parameter initialization
Wz = np.random.rand(h_size, N) * 0.1 - 0.05
Uz = np.random.rand(h_size, h_size) * 0.1 - 0.05
bz = np.zeros((h_size, 1))

Wr = np.random.rand(h_size, N) * 0.1 - 0.05
Ur = np.random.rand(h_size, h_size) * 0.1 - 0.05
br = np.zeros((h_size, 1))

Wh = np.random.rand(h_size, N) * 0.1 - 0.05
Uh = np.random.rand(h_size, h_size) * 0.1 - 0.05
bh = np.zeros((h_size, 1))

Wy = np.random.rand(o_size, h_size) * 0.1 - 0.05
by = np.zeros((o_size, 1))


 

# decode a one hot encoded string

 
# prepare data for the LSTM
def get_pair(n_in, n_out, n_unique):
	# generate random sequence
	sequence_in = generate_sequence(n_in, n_unique)
	sequence_out = sequence_in[:n_out] + [0 for _ in range(n_in-n_out)]
	# one hot encode

	return sequence_in,sequence_out
 
# generate random sequence
#inp,tar = get_pair(5, 2, 50)

#use to get the sample of the train
def sample(h, seed_inp, n):
    # Initialize first word of sample ('seed') as one-hot encoded vector.
    #turn the input to one_hot encoder
    one_hot_inp = one_hot_encode(seed_inp, vocab_size)
 
    ixes = []

    for t in range(len(x)):

        # Calculate update and reset gates
        #turn the one_hot_inp to (50,1) from (50,)
        one_hot = one_hot_inp[t].ravel()
        
        
        inp = []
        for i in one_hot:
  
            inp.append([i])

        inp = np.array(inp)

        z = sigmoid(np.dot(Wz, inp) + np.dot(Uz, h) + bz)
        r = sigmoid(np.dot(Wr, inp) + np.dot(Ur, h) + br)
        
        # Calculate hidden units
        h_hat = tanh(np.dot(Wh, inp) + np.dot(Uh, np.multiply(r, h)) + bh)
        h = np.multiply(z, h) + np.multiply((1 - z), h_hat)
        
        # Regular output unit
        y = np.dot(Wy, h) + by
        
        #Probability distribution

        e = softmax(y)

        #print(p.ravel())
        #print(len(p.ravel()))

        e=e.tolist()
        e=list(chain.from_iterable(e))

        # Choose next char according to the distribution
        ix = np.random.choice(range(50),p=e)
        
        ixes.append(ix)
  

        
        

    return ixes


def lossFun(inputs, targets, hprev):
    # Initialize variables
    z, r, h_hat, h, y, p =  {}, {}, {}, {-1: hprev}, {}, {} # Dictionaries contain variables for each timestep.
    sequence_loss = 0
    one_hot_inp = one_hot_encode(inputs, 50)
    
    one_hot_tar = one_hot_encode(targets,50)

    # Forward prop
    for t in range(len(one_hot_inp)):
        #the vocab_size =5
        # Set up one-hot encoded input and target 

        #turn the shape from (50,) to (50,1) 
        inp_one_hot = one_hot_inp[t].ravel()
        inp = []
        for i in inp_one_hot:
            inp.append([i])
        inp = np.array(inp)
        
        tar_one_hot = one_hot_tar[t].ravel()
        tar = []
        for i in tar_one_hot:
            tar.append([i])
        tar = np.array(tar)
     
        
        # Calculate update and reset gates
        z[t] = sigmoid(np.dot(Wz, inp) + np.dot(Uz, h[t-1]) + bz)
        r[t] = sigmoid(np.dot(Wr, inp) + np.dot(Ur, h[t-1]) + br)
        
        # Calculate hidden units
        h_hat[t] = tanh(np.dot(Wh, inp) + np.dot(Uh, np.multiply(r[t], h[t-1])) + bh)
        h[t] = np.multiply(z[t], h[t-1]) + np.multiply((1 - z[t]), h_hat[t])
        
        # Regular output unit
        y[t] = np.dot(Wy, h[t]) + by
        
        # Probability distribution
    
        p[t] = softmax(y[t])


    
        # Cross-entropy loss
        loss = -np.sum(tar*np.log(p[t]))
        sequence_loss += loss
        
        #- tf.reduce_sum(target * tf.log(output), axis)

    # Parameter gradient initialization
    dWy, dWh, dWr, dWz = np.zeros_like(Wy), np.zeros_like(Wh), np.zeros_like(Wr), np.zeros_like(Wz)
    dUh, dUr, dUz = np.zeros_like(Uh), np.zeros_like(Ur), np.zeros_like(Uz)
    dby, dbh, dbr, dbz = np.zeros_like(by), np.zeros_like(bh), np.zeros_like(br), np.zeros_like(bz)
    dhnext = np.zeros_like(h[0])
    
    # Backward prop
    for t in reversed(range(len(inputs))):
        # ∂loss/∂y
        dy = np.copy(p[t])
        dy[targets[t]] -= 1
        
        # ∂loss/∂Wy and ∂loss/∂by
        dWy += np.dot(dy, h[t].T)

        dby += dy
        
        # Intermediary derivatives
        dh = np.dot(Wy.T, dy) + dhnext
        dh_hat = np.multiply(dh, (1 - z[t]))
        dh_hat_l = dh_hat * tanh(h_hat[t], deriv=True)

        # ∂loss/∂Wh, ∂loss/∂Uh and ∂loss/∂bh
        dWh += np.dot(dh_hat_l, inp[t].T)
        dUh += np.dot(dh_hat_l, np.multiply(r[t], h[t-1]).T)
        dbh += dh_hat_l
        
        # Intermediary derivatives
        drhp = np.dot(Uh.T, dh_hat_l)
        dr = np.multiply(drhp, h[t-1])
        dr_l = dr * sigmoid(r[t], deriv=True)
        
        # ∂loss/∂Wr, ∂loss/∂Ur and ∂loss/∂br
        dWr += np.dot(dr_l, inp[t].T)
        dUr += np.dot(dr_l, h[t-1].T)
        dbr += dr_l
        
        # Intermediary derivatives
        dz = np.multiply(dh, h[t-1] - h_hat[t])
        dz_l = dz * sigmoid(z[t], deriv=True)
        
        # ∂loss/∂Wz, ∂loss/∂Uz and ∂loss/∂bz
        dWz += np.dot(dz_l, inp[t].T)
        dUz += np.dot(dz_l, h[t-1].T)
        dbz += dz_l
        
        # All influences of previous layer to loss
        dh_fz_inner = np.dot(Uz.T, dz_l)
        dh_fz = np.multiply(dh, z[t])
        dh_fhh = np.multiply(drhp, r[t])
        dh_fr = np.dot(Ur.T, dr_l)
        
        # ∂loss/∂h𝑡₋₁
        dhnext = dh_fz_inner + dh_fz + dh_fhh + dh_fr

    return sequence_loss, dWy, dWh, dWr, dWz, dUh, dUr, dUz, dby, dbh, dbr, dbz, h[len(inputs) - 1]

n, p = 0, 0
mdWy, mdWh, mdWr, mdWz = np.zeros_like(Wy), np.zeros_like(Wh), np.zeros_like(Wr), np.zeros_like(Wz)
mdUh, mdUr, mdUz = np.zeros_like(Uh), np.zeros_like(Ur), np.zeros_like(Uz)
mdby, mdbh, mdbr, mdbz = np.zeros_like(by), np.zeros_like(bh), np.zeros_like(br), np.zeros_like(bz)
smooth_loss = -np.log(1.0/vocab_size)*seq_length


print_interval = 100

while True:
    # Reset memory if appropriate
    if p + seq_length + 1 >= 1000 or n == 0:
        hprev = np.zeros((h_size, 1))
        p = 0
    
    # Get input and target sequence
    inputs,targets = get_pair(5, 2, 50)

    # Occasionally sample from model and print result
    if n % print_interval == 0:
        print("------------------------")
        print(inputs)
        sample_ix = sample(hprev, inputs, 1000)
        print(sample_ix)
        print("------------------------")

    # Get gradients for current model based on input and target sequences
    loss, dWy, dWh, dWr, dWz, dUh, dUr, dUz, dby, dbh, dbr, dbz, hprev = lossFun(inputs, targets, hprev)
    smooth_loss = smooth_loss * 0.999 + loss * 0.001

    # Occasionally print loss information
    if n % print_interval == 0:
        print('iter %d, loss: %f, smooth loss: %f' % (n, loss, smooth_loss))

    # Update model with adagrad (stochastic) gradient descent
    for param, dparam, mem in zip([Wy,  Wh,  Wr,  Wz,  Uh,  Ur,  Uz,  by,  bh,  br,  bz],
                                  [dWy, dWh, dWr, dWz, dUh, dUr, dUz, dby, dbh, dbr, dbz],
                                  [mdWy,mdWh,mdWr,mdWz,mdUh,mdUr,mdUz,mdby,mdbh,mdbr,mdbz]):
        np.clip(dparam, -5, 5, out=dparam)
        mem += dparam * dparam
        param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # Small added term for numerical stability

    # Prepare for next iteration
    p += seq_length
    n += 1


