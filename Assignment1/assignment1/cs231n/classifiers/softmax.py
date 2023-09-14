from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train = X.shape[0]  
    num_classes = W.shape[1]
    for i in range(num_train):
     score = X[i].dot(W)
     exp_score = np.exp(score)
     sum_exp_score = np.sum(exp_score)

     real_class = y[i]

     loss += (np.log(sum_exp_score) - score[real_class])

     dW[:, real_class] -= X[i]
     for j in range(num_classes):
      dW[:, j] += (X[i] * exp_score[j])/sum_exp_score

    loss /= float(num_train) 
    dW /= float(num_train)   # mini-batch SGD/average gradient

    loss += 0.5 * reg * np.sum(W*W)
    dW += reg * W 

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0]

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    scores = np.dot(X, W)   # (N, C)
    # print('shape of max 1: ', np.max(scores, axis=1).shape)
    # print('shape of max 2: ', np.max(scores, axis=1, keepdims=True).shape)
    exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True)) # (N,C)
    sum_exp_scores = np.sum(exp_scores, axis=1, keepdims=True) # (N, 1)
    probs = exp_scores / sum_exp_scores  # (N, C)
    correct_probs = np.choose(y, probs.T)

    loss = np.sum(-np.log(correct_probs))
    loss /= float(num_train)
    loss += 0.5 * reg * np.sum(W*W)

    dscores = probs.copy()
    dscores[np.arange(num_train), y] -= 1.0

    dW += np.dot(X.T, dscores)
    dW /= float(num_train)
    dW += reg * W


    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
