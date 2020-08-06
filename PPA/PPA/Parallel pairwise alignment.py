from multiprocessing import Pool
from time import sleep
import ppyClasses
import ppyFunctions

if __name__ == "__main__":
    # Read files and put them into arrays
    path = "C:\\Users\\netan\\Dropbox\\Research\\Cancer ML project\\PythonFiles\\Parallel pairwise alignment\\"
    file1 = "Test1-Pat1.txt"
    file2 = "Test1-Pat2.txt"
    numCores = 4
    gapScore = 0.4
    maxDel = 500  # An upper bound on the number of expected deletions. Set to -1 if not wanted.

    seq1 = []
    seq2 = []

    sequences = [[file1, seq1], [file2, seq2]]

    for seq in sequences:
        f = open(path + seq[0], 'r')
        for line in f:
            lineSplit = line.split(",")
            seq[1].append(ppyClasses.DataPoint(lineSplit[0], int(lineSplit[1]), int(lineSplit[2])))
        f.close()

    m1 = len(seq1)
    m2 = len(seq2)

    # Number of horizontal and vertical partitions.
    m1P = 2
    m2P = 2

    # Append dummy points to have m1P|m1 and m2P|m2
    dummyPoint = ppyClasses.DataPoint('QQ', -1, -1)

    while m1 % float(m1P) != 0:
        sequences[0][1].append(dummyPoint)
        m1 += 1
    while m2 % float(m2P) != 0:
        sequences[1][1].append(dummyPoint)
        m2 += 1

    v1 = m1 / m1P
    v2 = m2 / m2P

    # Initialize data structures.
    manager = ppyClasses.DPmanager()
    manager.start()
    DPdata = manager.DPdata(seq1, seq2, v1, v2, gapScore, m1P, m2P, maxDel)

    # Start algorithm.
    pool = Pool(numCores)

    debugCounter = 0

    # Manage the workers pool diagonal by diagonal
    for i in range(m1P + m2P - 1):
        avSlots = ppyFunctions.ithDiagonal(i, m1P, m2P, v1, v2, maxDel)
        results = [pool.apply_async(ppyFunctions.DPlocal,
                                    args=(DPdata.infoToSlot(avSlot), v1, v2,
                                          [s.get_pattern() for s in seq1[avSlot[0]*v1 : (avSlot[0] + 1)*v1]],
                                          [s.get_pattern() for s in seq2[avSlot[1]*v2 : (avSlot[1] + 1)*v2]],
                                          gapScore,
                                          avSlot,))
                   for avSlot in avSlots]
        [DPdata.retrieveResults(p.get()) for p in results]  # Wait for all processes to finish.
        print "Finished diagonal", i

    DPdata.printAlignment()
    print "Done."
