import operator
import copy

def editdist(str1, str2):
    # A safety feature to prevent alignment of real patterns to dummy patterns.
    if (str1 == 'QQ' and str2 != 'QQ') or (str2 == 'QQ' and str1 != 'QQ'):
        return 100
    m = len(str1)
    n = len(str2)
    v = [[0 for i in range(n+1)] for j in range(m+1)]  # Avoid using numpy.
    for i in range(m):
        v[i+1][0] = i
    for j in range(n):
        v[0][j+1] = j
    for i in range(m):
        for j in range(n):
            if str1[i] == str2[j]:
                v[i+1][j+1] = v[i][j]
            else:
                v[i+1][j+1] = 1 + min(min(v[i+1][j], v[i][j+1]), v[i][j])
    return v[m][n]


def DPlocal(peripheralValues, v1, v2, seq1, seq2, gapScore, slot):
    scoreSubmatrix = [[0 for i in range(v2+1)] for j in range(v1+1)]
    btSubmatrix = [['' for i in range(v2)] for j in range(v1)]

    scoreSubmatrix[0][0] = peripheralValues['dScore']
    for i in range(v1):
        scoreSubmatrix[i + 1][0] = peripheralValues['lScores'][i]
    for j in range(v2):
        scoreSubmatrix[0][j + 1] = peripheralValues['uScores'][j]

    for i in range(1, v1 + 1):
        for j in range(1, v2 + 1):
            # The score for current alignment:
            alignScore = editdist(seq1[i-1], seq2[j-1]) / \
                         ((len(seq1[i-1]) + len(seq2[j-1])) / float(2))
            dScore = scoreSubmatrix[i-1][j-1] + alignScore
                # DPdata.getScore(i - 1, j - 1) + alignScore  # The overall score for alignment.
            uScore = scoreSubmatrix[i-1][j] + gapScore
                # DPdata.getScore(i-1, j) + DPdata.get_gapScore()  # The overall score for left deletion.
            lScore = scoreSubmatrix[i][j-1] + gapScore
                #DPdata.getScore(i, j-1) + DPdata.get_gapScore()  # The overall score for right deletion.

            # Fill in the score matrix and the backtracking matrix.
            scoreSubmatrix[i][j] = min(uScore, lScore, dScore)
            tempdictionary = {'u': uScore, 'l': lScore, 'd': dScore}
            btSubmatrix[i-1][j-1] = min(tempdictionary.iteritems(), key=operator.itemgetter(1))[0]

    # Remove the top row and leftmost column.
    returnScoreMatrix = [[scoreSubmatrix[j + 1][i + 1] for i in range(v2)] for j in range(v1)]

    return returnScoreMatrix, btSubmatrix, slot


# The slot matrix has m1P+m2P-1 diagonals, each of which is a set of jobs that can happen concurrently.
# The following function returns the i-th diagonal, for 0 =< i =< m1P+m2P-2
# If maxDel != -1, limit the length of the diagonals to accelerate the algorithm.
def ithDiagonal(i, m1P, m2P, v1, v2, maxDel):
    start = (min(i, m1P - 1), max(0, i - m1P + 1))
    diag = [start]
    slotIter = copy.deepcopy(start)
    while slotIter[0] != 0 and slotIter[1] != m2P-1:
        slotIter = (slotIter[0] - 1, slotIter[1] + 1)
        # Artificial data restriction:
        # if abs((slotIter[0]+1)*v1-(slotIter[1]+1)*v2) <= maxDel or maxDel == -1:
        if abs(slotIter[0]*v1 - slotIter[1]*v2) <= maxDel or maxDel == -1:
            diag.append(copy.deepcopy(slotIter))
    return diag