# Name : Alok Kumar Pandey
# Matr No : 8654287
# IMT UserName : alok

import tensorflow as tf
import keras.backend as k

def gradeToProbability(grade,maxgrade):
    R = (k.pow(2,grade) - 1)/k.pow(2,maxgrade)
    return  R

def computeERR(relevanceGrades,pos):
    assert pos <= len(relevanceGrades),"pos is greater than the total no of relevance grades"
    grades = relevanceGrades[:pos]
    N = len(grades)
    grades = tf.convert_to_tensor(grades)
    maxgrade = k.max(grades)
    p = 1
    ERR = 0
    for r in range(1,N+1):
        R = gradeToProbability(grades[r-1],maxgrade)
        ERR = ERR + p * R/r
        p = p*(1-R)
    return ERR


if __name__ == '__main__':
    # graded relevance judgments
    # (e.g., 5:Perfect 4:satisfactory, 3:Highly relevant, 2:relevant, 1: marginal, 0: irrelevant)
    relevanceGrades = [3,4,2,2,1]
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        pos = 3
        ERR = sess.run(computeERR(relevanceGrades,pos))
        print ('ERR at position {} is: {} given relevance grades: {}'.format(pos,ERR,relevanceGrades))
        assert ERR == 0.703369140625,"ERR calculation is wrong for {}".format(pos)

        # pos = 5
        # ERR = sess.run(computeERR(relevanceGrades, pos))
        # print('ERR at position {} is: {} given relevance grades: {}'.format(pos, ERR, relevanceGrades))
        # assert ERR == 0.703369140625, "ERR calculation is wrong for {}".format(pos)

        pos = 7
        ERR = sess.run(computeERR(relevanceGrades, pos))
        print('ERR at position {} is: {} given relevance grades: {}'.format(pos, ERR, relevanceGrades))
        assert ERR == 0.703369140625, "ERR calculation is wrong for {}".format(pos)
