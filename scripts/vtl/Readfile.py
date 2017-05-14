import numpy as np

def Readfile(f, h=1, is_float=True, seperator=' '):
    header = []
    column = []

    #####open file to find length of columns
    y = open(f, 'r')
    for i in range(h):
        d = y.readline()
    columns = y.readline().split(seperator)
    c = len(columns)
    y.close()

    #####initialize columns and rows
    for i in range(c):
        column.append([i])

    #####read in header
    f = open(f, 'r')
    for i in range(h):
        header.append(f.readline())

    #####Create columns and rows
    for line in f:
        line = line.strip()
        columns = line.split(seperator)
        p = columns
        for u in range(c):

            if is_float is True:
                column[u].append(float(p[u]))
            else:
                column[u].append(p[u])

    #####remove first entries
    f.close()
    for i in range(c):
        column[i].remove(i)

    #####return arrays
    return np.array(column)