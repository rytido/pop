import numpy as np
import pyaudio
import whisper

RATE = 16000 # 16K samples per second
CHUNK = RATE * 5 # read for 5 seconds

model = whisper.load_model("base.en", device="cpu")

if __name__ == '__main__':
  p = pyaudio.PyAudio()
  stream = p.open(input=True, format=pyaudio.paInt16, channels=1, rate=RATE, frames_per_buffer=1024)

  while True:
    data = stream.read(CHUNK)
    x = np.frombuffer(data, np.int16).astype(np.float32).flatten()/32768.0
    result = model.transcribe(x, language="en")
    print(result)
    break
