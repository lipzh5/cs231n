from builtins import range
from urllib.request import parse_keqv_list
from matplotlib.collections import JoinStyle
from matplotlib.font_manager import json_load
import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]  # num of minibatch size
    loss = 0.0
    N = X.shape[0]
    _, C = W.shape
    for i in range(N):
      scores = X[i].dot(W)  # (1, C)
      s_yi = scores[y[i]]
      for j in range(C):
        if j == y[i]:
          continue
        margin = scores[j] - s_yi + 1
        if margin > 0:
          loss += margin
          dW[:,j] += X[i]
          dW[:,y[i]] -= X[i]
    loss /= float(N)
    dW /= float(N)
  
    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
   
    # Add L2 regularization to the loss.
    loss += 0.5 * reg * np.sum(W**2)
    dW += reg * W
   
    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N, D = X.shape
    scores = X.dot(W)  # (N, C)
    # choose one (indexed by y) per column 
    correct_class_scores = np.choose(y, scores.T).reshape(-1, 1)  # (N, 1)
    # print('shape of correct lass scores: ', correct_class_scores.shape)
    margins = np.maximum(scores - correct_class_scores + 1, 0.0)
    margins[np.arange(N), y] = 0.0
    loss += np.sum(margins)/float(N)
    loss += 0.5 * reg * np.sum(W**2)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    grad_mask = (margins > 0).astype(int)
    grad_mask[np.arange(N), y] = -np.sum(grad_mask, axis=1)
    dW = np.dot(X.T, grad_mask)/float(N)
    dW += reg * W
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
