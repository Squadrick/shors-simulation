import numpy as np
import matplotlib.pyplot as plt

def Hadamard(n):
    def Hn(H=np.array([[1, 1], [1, -1]], dtype=np.complex64), n=n):
        if n > 1:
            return Hn(H=np.kron(np.array([[1, 1], [1, -1]], dtype=np.complex64), H), n=n-1)
        return H

    return Hn(n=n)

def QFT(t):
    Q = np.zeros(shape=(2 ** t, 2 ** t), dtype=np.complex64)
    N = 2 ** t
    for i in range(N):
        for j in range(N):
            Q[i][j] = np.exp(np.pi * 2j * ((i * j) % N) / N)

    return Q

N = 21
t = 9
H = Hadamard(t)

reg1 = np.zeros(shape=(2 ** t), dtype=np.complex64)
reg2 = np.ones(shape=(2 ** t), dtype=np.complex64)
reg1[0] = 1
reg1 = H.dot(reg1)

for i in range(2 ** t):
    reg2[i] = 2 ** i % N

r = reg2[0]

for i in range(2 ** t):
    if reg2[i] != r:
        reg1[i] = 0

Q = QFT(9)
reg1 = np.linalg.inv(Q).dot(reg1)

fig, ax = plt.subplots( nrows=1, ncols=1 )
ax.plot(abs(reg1))
fig.savefig('plot.png')
plt.close(fig)
