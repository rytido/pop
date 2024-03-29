import sys
from time import sleep, time
import struct
import numpy as np
import pyaudio
import wave

FPS = 40.0
window_time = 4
min_period = 0.2
factor_above = 1.8
factor_below = .7
nFFT = 128
BUF_SIZE = 4 * nFFT
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
xsl = int(FPS * window_time - 1)
chunk=4096
freq_window_pct = .2
l = int(nFFT/2 * (1-freq_window_pct))
u = int(nFFT/2 * (1+freq_window_pct))


def detect(stream, MAX_y):
  # Read n*nFFT frames from stream, n > 0
  N = int(max(stream.get_read_available() / nFFT, 1) * nFFT)
  data = stream.read(N)

  # Unpack data, LRLRLR...
  y = np.array(struct.unpack("%dh" % (N * CHANNELS), data)) / MAX_y
  y_L = y[::2]
  y_R = y[1::2]

  Y_L = np.fft.fft(y_L, nFFT)
  Y_R = np.fft.fft(y_R, nFFT)

  # Sewing FFT of two channels together, DC part uses right channel's
  Y = abs(np.hstack((Y_L[int(-nFFT / 2):-1], Y_R[:int(nFFT / 2)])))
  x = Y[l:u].mean()
  return x


def main(stream, out_stream, data):
  # Used for normalizing signal. If usout_streame paFloat32, then it's already -1..1.
  # Because of saving wave, paInt16 will be easier.
  MAX_y = 2.0 ** (p.get_sample_size(FORMAT) * 8 - 1)
  xs = np.array([])
  t0 = time()
  ok_to_trigger = False

  while True:
    x = detect(stream, MAX_y)
    xs = np.concatenate(([x], xs[:xsl]))
    t = time()
    if xs.shape[0] > 50:
      m = xs.mean()
      if not ok_to_trigger and t - t0 > min_period and x < m * factor_below:
        print("OK TO TRIGGER")
        ok_to_trigger = True

      if ok_to_trigger and x > m * factor_above + .1:
        out_stream.write(data)
        print(x, m)
        t0 = time()
        ok_to_trigger = False

    sleep(1/FPS)


if __name__ == '__main__':
  wf = wave.open('pop.wav', 'rb')
  dump = wf.readframes(512)
  data = wf.readframes(chunk)

  p = pyaudio.PyAudio()

  out_stream = p.open(format = p.get_format_from_width(wf.getsampwidth()),
                channels = wf.getnchannels(),
                rate = wf.getframerate(),
                output = True)

  stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=BUF_SIZE)

  try:
    main(stream, out_stream, data)
  except KeyboardInterrupt:
    print('Exiting')
    wf.close()
    stream.stop_stream()
    stream.close()
    out_stream.close()
    p.terminate()
    sys.exit(0)
