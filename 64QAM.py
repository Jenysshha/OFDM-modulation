import numpy as np
import matplotlib.pyplot as plt


n_fft = 1500
Q = 6
b = np.random.randint(0, 2, size = n_fft * Q)

#Модуляция

signal = ((1-2*b[::6])*(4-(1-2*b[2::6])*(2-(1-2*b[4::6])))+1j*(1-2*b[1::6])*(4-(1-2*b[3::6])*(2-(1-2*b[5::6]))))*(1/np.sqrt(42))

#ОБПФ

opf = np.fft.ifft(signal, norm='ortho')

#Шум

snr = 1000
power_signal = 1
power_noise = power_signal/snr
average = 0
standart_deviation = np.sqrt(power_noise)
num_samples = len(opf)

noise_re = np.random.normal(average,standart_deviation,size=num_samples) 
noise_im = np.random.normal(average,standart_deviation,size=num_samples) 

noise = noise_re+1j*noise_im
noise *= 1/np.sqrt(2)

#Сложение шума и ОПФ

noisy_opf = opf + noise

#ППФ
modulated = np.fft.fft(noisy_opf, norm='ortho')
plt.figure()
plt.plot(np.real(modulated),np.zeros(len(modulated)), color='black', linewidth=0.2)
plt.plot(np.zeros(len(modulated)),np.imag(modulated), color='black', linewidth=0.2)
plt.scatter(np.real(modulated),np.imag(modulated), s = 15)
#plt.ylim(-2.0,2.0)
#plt.xlim(-2.0,2.0)

plt.show()




