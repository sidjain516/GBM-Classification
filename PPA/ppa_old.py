from multiprocessing import Pool
import ppyClasses
import ppyFunctions
import sys
chromnum = sys.argv[3]
if __name__ == "__main__":
    # Read files and put them into arrays
    path = "/home/sjain2/PPA/"
    file1 = "/home/sjain2/chrs/chr"+chromnum+"_data.txt"
    #file1 = "Test1-Pat1.txt"
    file2 = sys.argv[1]
    numCores = 4
    gapScore = 0.4
    maxDel = 500  # An upper bound on the number of expected deletions. Set to -1 if not wanted.
    m1P = 100  # Number of horizontal partitions.
    m2P = 100  # Number of vertical partitions.

    seq1 = []
    seq2 = []

    sequences = [[file1, seq1], [file2, seq2]]

    #for seq in sequences:
    f = open(file1, 'r')
    for line in f:
        lineSplit = line.split(" ")
        seq1.append(ppyClasses.DataPoint(lineSplit[14], int(lineSplit[16]), int(lineSplit[17])))
    f.close()
    f = open(file2, 'r')
    for line in f:
        lineSplit = line.split(" ")
        seq2.append(ppyClasses.DataPoint(lineSplit[14], int(lineSplit[16]), int(lineSplit[17])))
    f.close()
    m1 = len(seq1)
    m2 = len(seq2)

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
