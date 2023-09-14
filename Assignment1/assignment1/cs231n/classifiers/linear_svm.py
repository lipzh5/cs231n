from builtins import range
from urllib.request import parse_keqv_list
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
    # pz: y is the label indicator, y[i]=c, means X[i] has label c
    # X is of shape (500, 3073)
    for i in range(num_train):
      scores = X[i].dot(W)  # (1, 10)
      real_class_score = scores[y[i]]
      for j in range(num_classes): # calculate loss
        if j == y[i]:
          continue
        margin = scores[j] - real_class_score + 1
        if margin > 0:
          loss += margin
          dW[:, j] += X[i]
          dW[:, y[i]] -= X[i]
   
    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= float(num_train)
    dW /= float(num_train)

    # Add L2 regularization to the loss.
    loss += 0.5 * reg * np.sum(W*W)
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
    # scores = W.dot(X)
    # print('shape W', W.shape)
    # print('X is None? ', X is None)
    # print('shape X ', X.shape)
    # print('shape y', y.shape)
    # scores = X.dot(W)  
    # print('shape scores: ', scores.shape) # 500 *10
    scores = np.dot(X, W)  #(500, 10)
    # print('shape of scores: ', scores.shape)
    # np.choose: choose one per column
    correct_class_scores = np.choose(y, scores.T).reshape(-1,1)
    #correct_class_scores.reshape(-1,1)  # (500,1)
    # print('shape of correct class scores', correct_class_scores.shape)
    margins = np.maximum(scores - correct_class_scores + 1, 0.0)
    # print('shape of margins: ', margins.shape)  # (500, 10)
    num_train = X.shape[0]
    # print('np arange shape: ', np.arange(num_train).shape)
    margins[np.arange(num_train),y] = 0.0   # exclude the real class

    loss += np.sum(margins)/float(num_train)
    loss += 0.5 * reg * np.sum(W*W)  # L2 regularization



    # test np
    # a = np.array([[2,3],[1,0]]) 
    # b=np.array([1, -1])
    # b = b.reshape(-1,1)
    # print('a-b: ', a-b + 1)
    # # test np choose
    # choices = np.array([0,2,1])
    # print('shape of choices: ', choices.shape) # (3,) pick one per column
    # arr = np.array([[0,3,6],[1,4,7],[2,5,8]]) 
    # idxes_per_row = np.array([1, 0, 2])
    # arr[np.arange(3), idxes_per_row] = 0.0
    # print('arr changed111 ', arr)
    # arr[1, 0] = 2
    # print('arr changed 222: ', arr)
    # print('shape of arr: ', arr.shape)  # (3,3)
    # selected = np.choose(choices, arr)
    # print('selected: ', selected)
    # print('selected 2: ', np.choose(choices.reshape(1,-1), arr))
  
   

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

    grad_mask = (margins > 0).astype(int)   # binary value (500, 10)

    grad_mask[np.arange(y.shape[0]), y] = -np.sum(grad_mask, axis=1)

    dW = np.dot(X.T, grad_mask)
    dW/=float(num_train)

    dW += reg * W



    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
