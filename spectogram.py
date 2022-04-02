from scipy import signal
import matplotlib.pyplot as plt


def create_spectrogram(measured_signal, name, train_data_setup):
    f, t, Sxx = signal.spectrogram(measured_signal, train_data_setup.length_of_signal)
    plt.pcolormesh(t, f, Sxx, shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.savefig('spectrograms/' + name + '.png')
    plt.close()
