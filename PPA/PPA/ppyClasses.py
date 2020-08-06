from multiprocessing.managers import BaseManager
from scipy.sparse import dok_matrix
import numpy as np
class DataPoint:
    def __init__(self, pattern = 'N', mutationRate = -1, duplicationNum = -1):
        self.mutationRate = mutationRate
        self.duplicationNum = duplicationNum
        self.pattern = pattern

    def __str__(self):
        return "(%s, %d, %d)" % (self.pattern, self.mutationRate, self.duplicationNum)

    def get_pattern(self):
        return self.pattern


class DPdataClass:
    def __init__(self, seq1, seq2, v1, v2, gapScore, m1P, m2P, maxDel):
        self.seq1 = seq1
        self.seq2 = seq2
        self.v1 = v1
        self.v2 = v2
        self.m1P = m1P
        self.m2P = m2P

        #  self.scoreMatrix = [[0 for i in range(len(seq2)+1)] for j in range(len(seq1)+1)]  # Avoid using numpy.
        self.scoreMatrix = dok_matrix((len(seq1)+1, len(seq2)+1),dtype=np.float32)

        #  self.btMatrix = [[0 for i in range(len(seq2)+1)] for j in range(len(seq1)+1)]  # Avoid using numpy.
        self.btMatrix = dok_matrix((len(seq1)+1, len(seq2)+1),dtype='string')

        self.gapScore = gapScore
        self.maxDel = maxDel

        # Fill in the top rows and leftmost columns of scoreMatrix and btMatrix
        for i in range(1, len(seq1)+1):
            self.scoreMatrix[i, 0] = (i-1) * self.gapScore
            self.btMatrix[i, 0] = 'u'
        for j in range(1, len(seq2)+1):
            self.scoreMatrix[0, j] = (j-1) * self.gapScore
            self.btMatrix[0, j] = 'l'
        self.scoreMatrix[0, 0] = 0
        self.btMatrix[0, 0] = -1

    def printAlignment(self):
        seq1Aligned = []
        seq2Aligned = []

        delString = '-'

        t = (-1, -1)
        while self.btMatrix[t[0], t[1]] != -1:
            if self.btMatrix[t[0], t[1]] == 'l':
                seq1Aligned = [delString] + seq1Aligned
                seq2Aligned = [(self.seq2[t[1]]).__str__()] + seq2Aligned
                t = (t[0], t[1]-1)
            if self.btMatrix[t[0], t[1]] == 'u':
                seq1Aligned = [(self.seq1[t[0]]).__str__()] + seq1Aligned
                seq2Aligned = [delString] + seq2Aligned
                t = (t[0]-1, t[1])
            if self.btMatrix[t[0], t[1]] == 'd':
                seq1Aligned = [(self.seq1[t[0]]).__str__()] + seq1Aligned
                seq2Aligned = [(self.seq2[t[1]]).__str__()] + seq2Aligned
                t = (t[0]-1, t[1]-1)

        # Print to file "alignment.txt".
        outputFile = open("alignment.txt", 'w')
        printstr = []
        for s in seq1Aligned:
            if s[1:3] == 'QQ':
                break
            printstr.append("{0:20}".format(s))
        # print printstr
        outputFile.writelines(printstr)
        outputFile.writelines('\n')
        printstr = []
        for s in seq2Aligned:
            if s[1:3] == 'QQ':
                break
            printstr.append("{0:20}".format(s))
        # print printstr
        outputFile.writelines(printstr)
        outputFile.close()

    def infoToSlot(self, slot):
        # Calculate the upper left corner indices.
        ulCorner1 = 1 + slot[0]*self.v1
        ulCorner2 = 1 + slot[1]*self.v2

        return {'dScore': self.scoreMatrix[ulCorner1 - 1, ulCorner2 - 1],
                'uScores': [self.scoreMatrix[ulCorner1 - 1, ulCorner2 + j] for j in range(self.v2)],
                'lScores': [self.scoreMatrix[ulCorner1 + j, ulCorner2 - 1] for j in range(self.v1)]}

    def retrieveResults(self, res):
        scoreSubmatrix = res[0]
        btSubmatrix = res[1]
        slot = res[2]

        for i in range(self.v1):
            for j in range(self.v2):
                self.scoreMatrix[slot[0]*self.v1 + 1 + i, slot[1]*self.v2 + 1 + j] = scoreSubmatrix[i][j]
                self.btMatrix[slot[0] * self.v1 + 1 + i, slot[1] * self.v2 + 1 + j] = btSubmatrix[i][j]

    def putScore(self, i, j, val):
        self.scoreMatrix[i, j] = val

    def putBT(self, i, j, val):
        self.btMatrix[i, j] = val

    def get_seq1(self, i):
        return (self.seq1[i]).get_pattern()

    def get_seq2(self, i):
        return (self.seq2[i]).get_pattern()

    def getScore(self, i, j):
        # Artificial restriction of the data:
        if abs(i-j) >= self.maxDel:
            return 10**6
        return self.scoreMatrix[i, j]

    def getBT(self, i, j):
        # Artificial restriction of the data:
        if i-j > self.maxDel:
            return 'u'
        if j-i > self.maxDel:
            return 'l'
        return self.btMatrix[i, j]

    def get_v1(self):
        return self.v1

    def get_v2(self):
        return self.v2

    def get_gapScore(self):
        return self.gapScore

class DPmanager(BaseManager):
    pass


DPmanager.register('DPdata', DPdataClass)
