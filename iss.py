import numpy as np
import matplotlib.pyplot as plt
import IPython
import math
from numpy.core.numeric import isclose
from scipy.signal import spectrogram, lfilter, freqz, tf2zpk, find_peaks
from scipy.io import wavfile

# plt.style.use("seaborn-whitegrid")


def dft(array):
    # array = np.append(array, np.zeros(1024 - len(array)))
    # arr = []
    # for i in range(len(array)):
    #     coef = 0
    #     for j in range(len(array)):
    #         coef += array[j] * np.exp(-2j * np.pi * i * j * (1 / len(array)))
    #     # ---------------------------------------------------------------zapisujem abs
    #     arr.append(coef)
    # return arr
    N = len(array)
    n = np.arange(N)
    k = n.reshape((N, 1))
    e = np.exp(-2j * np.pi * k * n / N)

    X = np.dot(e, array)

    return X


###########################################################
# Zakladny signal
fs, data = wavfile.read("audio/example.wav")
t = np.arange(data.size) / fs
plt.figure(1)
plt.plot(t, data)
plt.xlabel("t[s]")
plt.ylabel("Amplituda")
plt.title("Zvukovy signal")

print("Minimalna hodnota : ", data.min())
print("Maximalna hodnota : ", data.max())
print("Dlzka nahravky(vo vzorkoch) : ", len(data))
print("Dlzka nahravky(v sekundach) : ", len(data) / fs)
print("Vzorkovacia frekvencia : ", fs)
###########################################################

###########################################################
# Odstranenie jednosmernej zlozky
for n in range(0, data.size):
    data[n] = data[n] - np.mean(data)
###########################################################

###########################################################
# Normalizacia
data = data / max(data.max(), -data.min())
plt.figure(2)
plt.title("Normalizovany signal")
plt.xlabel("t(s)")
plt.ylabel("Amplituda")
plt.plot(t, data)
###########################################################

###########################################################
# Ramcovanie
n = 1024  # velkost ramca
m = 512  # dlzka prekrytia
array = [data[i : i + n] for i in range(0, len(data), n - m)]
index = 1  # index matice na vykreslenie
plt.figure(3)
plt.plot(np.arange(index * 512, index * 512 + 1024) / fs, array[index])
plt.title("Ramec signalu")
plt.xlabel("t(s)")
plt.ylabel("Amplituda")
###########################################################

###########################################################
# DFT
after_dft = dft(array[index])
plt.figure(4)
y = np.arange(0, 1024 / 2) * (fs / 2048 * 2)  # pre 512 vzorkov rozsah od 0Hz do 8kHz
plt.plot(y, np.abs(after_dft[:512]))
plt.title("DFT")
plt.xlabel("f(Hz)")
# plt.ylabel("Amplituda")
###########################################################

###########################################################
# Kontrola DFT
correctDFT = np.fft.fft(array[index])
# plt.figure(9)
# plt.plot(np.arange(0, 1024) * (fs / 2048), correctDFT)
# plt.title("DFT gen")
# plt.xlabel("f(Hz)")
# plt.figure(10)
# plt.plot(np.arange(0, 1024) * (fs / 2048), after_dft)
# plt.title("DFT moja")
# plt.xlabel("f(Hz)")
if np.allclose(after_dft, correctDFT):
    print("DFTs match")
else:
    print("Something is wrong")
###########################################################

###########################################################
# Spektogram
after_log = 10 * np.log10(np.abs(after_dft) ** 2)
plt.figure(5)
plt.plot(np.arange(0, 1024) * (fs / 2048), after_log)
plt.title("DFT")
plt.xlabel("f(Hz)")


# plt.figure(figsize=(11, 5))
# plt.imshow(np.rot90(after_log), extent=[0, 1, 0, 8000], aspect='auto')
# plt.gca().set_title('DFT - Spektrogram bez roušky')
# plt.gca().set_xlabel('Čas [s]')
# plt.gca().set_ylabel('Frekvence [Hz]')
# cbar = plt.colorbar()
# cbar.set_label('Spektralní hustota výkonu [dB]', rotation=270, labelpad=15)


# plt.figure(figsize=(9, 3))
# # plt.pcolormesh(t, np.arange(0, 1024)*(fs/2048), after_log)
# plt.gca().set_xlabel('Čas [s]')
# plt.gca().set_ylabel('Frekvence [Hz]')
# cbar = plt.colorbar()
# cbar.set_label('Spektralní hustota výkonu [dB]', rotation=270, labelpad=15)
###########################################################

###########################################################
# Zistenie rusivych frekvencii
after_dft_abs = np.abs(after_dft)
peaks, _ = find_peaks(after_dft_abs[: len(after_dft) // 2], height=1)
peaks = peaks * (fs / 1024)
peaks = [np.floor(float(x)) for x in peaks]
print("Rusive frekvencie : ", peaks)
###########################################################

###########################################################
# Kontrola zistenych rusivych frekvencii
peak = peaks[0]
before_peak = peaks[0]
diff = 15
for i in peaks:
    if math.isclose(peak, i, abs_tol=diff):
        print("Peak ", i, "is OK")
    else:
        print("Peak ", i, "is NOT OK")
    peak += before_peak
    diff += 15
###########################################################

F = 600
T = 1 / F
Ts = 1.0 / fs
N = int(T / Ts)

t = np.linspace(0, len(data))
signal = np.cos(2 * np.pi * t)

time = np.arange(0, len(data), 1)
amplitude = np.cos(time * 2 * np.pi)

arr = []
for i in range(0, len(time)):
    item = 0
    for y in range(1, 5):
        item += np.cos(2 * np.pi * i / len(time) * y * peaks[0] * 2)
    arr.append(item)

plt.figure(6)

plt.plot(time, arr)
# arr = np.asarray(arr, dtype=np.int16)


correctDFT = np.fft.fft(arr)
plt.figure(7)
plt.plot(np.arange(0, len(arr)) / 3.3, correctDFT)
plt.title("DFT")
plt.xlabel("f(Hz)")


# sorted = np.sort(after_dft)
# plt.figure(6)
# plt.plot(np.arange(0, 1024)*(fs/2048), sorted)
# plt.title("DFT")
# plt.xlabel("f(Hz)")
# s_seg_spec = np.fft.fft(array[index])
# G = s_seg_spec

# # np.arange(n) vytváří pole 0..n-1 podobně jako obyč Pythonovský range
# # ax[0].plot(np.arange(s_seg.size) / fs, s_seg)
# # ax[0].set_xlabel('$t[s]$')
# # ax[0].set_title('Segment signalu $s$')
# # ax[0].grid(alpha=0.5, linestyle='--')

# #f = np.arange(G.size) / N * fs
# # zobrazujeme prvni pulku spektra
# plt.figure(4)

# plt.plot(s_seg_spec)
# # plt.set_xlabel('$f[Hz]$')
# # plt.set_title('Spektralni hustota vykonu [dB]')
# # plt.grid(alpha=0.5, linestyle='--')

# plt.tight_layout()

# dftted = dft(array[index])
# plt.figure(5)

# plt.plot(dftted)
# # plt.set_xlabel('$f[Hz]$')
# # plt.set_title('Spektralni hustota vykonu [dB]')
# # plt.grid(alpha=0.5, linestyle='--')

# plt.tight_layout()
# print(s_seg_spec[10], dftted[10])
# # odkud = 0.0  # 1.5     # začátek segmentu v sekundách
# # kolik = 1024  # délka segmentu v sekundách
# odkud_vzorky = int(odkud * fs)         # začátek segmentu ve vzorcích
# pokud_vzorky = int(odkud * fs + kolik)  # konec segmentu ve vzorcích

# s_seg = data[odkud_vzorky:pokud_vzorky]
# N = s_seg.size

# s_seg_spec = np.fft.fft(s_seg)
# plt.figure(3)
# plt.plot(np.arange(s_seg.size) / fs + odkud, s_seg)
# plt.xlabel("t[s]")
# plt.title("Segment signalu")
# plt.grid(alpha=0.5, linestyle='--')

f, t, sgr = spectrogram(data, fs)
sgr_log = 10 * np.log10(abs(sgr + 1e-20) ** 2)
plt.grid(False)
plt.figure(figsize=(9, 3))
plt.pcolormesh(t, f, sgr_log)
plt.gca().set_xlabel("Čas [s]")
plt.gca().set_ylabel("Frekvence [Hz]")
cbar = plt.colorbar()
cbar.set_label("Spektralní hustota výkonu [dB]", rotation=270, labelpad=15)


arr = np.asarray(arr, dtype=np.int16)

f, t, sgr = spectrogram(arr, fs)
# prevod na PSD
# (ve spektrogramu se obcas objevuji nuly, ktere se nelibi logaritmu, proto +1e-20)
sgr_log = 10 * np.log10(abs(sgr + 1e-20) ** 2)
plt.grid(False)
plt.figure(figsize=(9, 3))
plt.pcolormesh(t, f, sgr_log)
plt.gca().set_xlabel("Čas [s]")
plt.gca().set_ylabel("Frekvence [Hz]")
cbar = plt.colorbar()
cbar.set_label("Spektralní hustota výkonu [dB]", rotation=270, labelpad=15)

# plt.tight_layout()
# plt.tight_layout()
plt.show()
