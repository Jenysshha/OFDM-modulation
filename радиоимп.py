# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 13:17:58 2023

@author: User
"""

import numpy as np
import matplotlib.pyplot as plt

T = 1
N = 2048
A = 1.0

L = 2.0

t = np.linspace(-L / 2, +L / 2, N, endpoint=False)

x = np.zeros(t.shape)
x[(-T / 2 <= t) & (t <= T / 2)] = A

f1=5 #частота заполнения нашего видеоимпульса
x1=np.cos(2*np.pi*f1*t)

l=len(x)
X=[]
i=0
while i<l:
    X.append(x[i]*x1[i])
    i+=1


yf=np.fft.fft(x, norm='ortho')
yf=np.fft.fftshift(yf)

F = N / L
df = 1 / L
f = np.arange(-F / 2, + F / 2, df)

Xf=np.fft.fft(X, norm='ortho')
Xf=np.fft.fftshift(Xf)

fig = plt.figure()

pf = abs(yf) ** 2
pf_dB = 10 * np.log10(pf / pf.max())

ax=fig.add_subplot(2,2,1)
ax1=fig.add_subplot(2,2,2)
ax2=fig.add_subplot(2,2,3)
ax3=fig.add_subplot(2,2,4)

plt.xlim([-100, 100])

ax.plot(t, x)
ax2.plot(f, yf)

ax1.plot(t,X)
ax3.plot(f,Xf)


plt.show()