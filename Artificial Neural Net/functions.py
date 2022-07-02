import numpy


# Activation functions for the Algorithm
def sigmoid(Z) :
    s = 1 / (1 + numpy.exp(-Z))
    return s

def relu(Z) :
    r = numpy.maximum(0, Z)
    return r

def d_sigmoid(Z) :
    gZ = sigmoid(Z)
    d_s = gZ * (1 - gZ)
    return d_s

def d_relu(Z) :
    d_r = numpy.where(Z > 0, 1, 0)
    return d_r

def tanh(Z) :
    t = numpy.tanh(Z)
    return t

def d_tanh(Z) :
    d_t = 1 - tanh(Z) ** 2
    return d_t

# Handle Activation and Deactivation for the functions
def activate(Z, mode):

    if mode == 'sigmoid':
        return sigmoid(Z)

    elif mode == 'relu':
        return relu(Z)

    elif mode == 'tanh':
        return tanh(Z)

def d_activate(Z, mode):

    if mode == 'sigmoid':
        return d_sigmoid

    elif mode == 'relu':
        return d_relu(Z)

    elif mode == 'tanh':
        return d_tanh(Z)

