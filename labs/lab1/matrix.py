import numpy as np

def dot(X, Y):
    r = 0
    for i in range(len(X)):
        r += X[i]*Y[i]
    return r


def matmul(A, B):
    m = len(A)
    n = len(B[0])
    result = [[0 for _ in range(n)] for _ in range(m)] # create zeros
    for i in range(m):
        for j in range(n):
            col = [B[x][j] for x in range(m)]
            result[i][j] = dot(A[i], col)

    print(result)

A = [[1, 1], [0, 1]]
B = [[2, 3], [1, 2]]
matmul(A, B)

print(np.array(A) @ np.array(B))