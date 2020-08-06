import sys

with open(sys.argv[1]) as f:
    file = f.readlines()
out = open(sys.argv[2], 'w')

for line in file:
    if line[0] != '-':
        if line[30] == '-':
            out.write('0 0 \n')
        else:
            s = line[30:]
            dat = s.strip().split(", ")
            dat2 = dat[1:]
            out.write(dat2[0] + ' ')
            out.write(dat2[1][:-1] + '\n')
            
