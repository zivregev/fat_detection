# from num import integrate_f
#
# if __name__ == "__main__":
#     print (integrate_f(1.0, 10.0, 2000))

import numpy as np


class UnionFind:
    def __init__(self):
        self._ids = []
        self._sizes = []

    def make_set(self):
        self._ids.append(len(self._ids) + 1)
        self._sizes.append(1)
        return self._ids[-1]

    def root(self, i):
        while i != self._ids[i]:
            self._ids[i] = self._ids[self._ids[i]]
            i = self._ids[i]
        return i

    def find(self, p):
        return self._ids[p]

    def union(self, p, q):
        i = self.root(p)
        j = self.root(q)
        if self._sizes[i] < self._sizes[j]:
            self._ids[i] = j
            self._sizes[j] += self._sizes[i]
            return j
        else:
            self._ids[j] = i
            self._sizes[i] += self._sizes[j]
            return i


def label_array(arr):
    labels = np.copy(arr)
    uf = UnionFind(np.count_nonzero(arr))
    n, m = arr.shape
    for i in range(n):
        for j in range(m):
            if labels[i][j]:
                up = 0 if i == 0 else labels[i-1, j]
                left = 0 if j == 0 else labels[i, j-1]
                composite = sum(map(lambda x: bool(x)))
                if composite == 0:
                    labels[i, j] = uf.make_set()
                elif composite == 1:
                    labels[i, j] = max(up, left)
                else:
                    labels[i, j] = uf.union(up, left)