import numpy as np
import pyaudio
from queue import Queue
import threading
from time import sleep
import whisper

RATE = 16000  # 16K samples per second
CHUNK = RATE
MIN_CHUNKS = 3
MAX_CHUNKS = 10
START_THRESHOLD = 0.0004
END_THRESHOLD = 0.0004


def empty_and_sound(rms, n):
    return rms > START_THRESHOLD and n == 0


def non_empty_but_short(n):
    return n > 0 and n < MIN_CHUNKS


def not_long_and_sound(rms, n):
    return rms > END_THRESHOLD and n < MAX_CHUNKS


queue = Queue(maxsize=10)


def producer():
    p = pyaudio.PyAudio()
    stream = p.open(
        input=True,
        format=pyaudio.paInt16,
        channels=1,
        rate=RATE,
        frames_per_buffer=1024,
    )

    segment = []

    while True:
        data = stream.read(CHUNK)
        x = np.frombuffer(data, np.int16).astype(np.float32) / 32768.0
        rms = np.sqrt(np.mean(x**2))
        n = len(segment)
        print(rms)
        if (
            empty_and_sound(rms, n)
            or non_empty_but_short(n)
            or not_long_and_sound(rms, n)
        ):
            segment.append(x)
        elif n > 0:
            print("ADDING SEGMENT TO QUEUE", n)
            audio = np.concatenate(segment)
            queue.put(audio)
            segment = []


model = whisper.load_model("base.en", device="cpu")


def consumer():
    while True:
        audio = queue.get()
        result = model.transcribe(audio, fp16=False, language="en")
        print(result)
        queue.task_done()
        sleep(0.01)


if __name__ == "__main__":
    producer_thread = threading.Thread(target=producer)
    consumer_thread = threading.Thread(target=consumer)

    producer_thread.start()
    consumer_thread.start()
