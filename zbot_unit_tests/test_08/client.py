"""Client for the voice2action on Zbot."""
import argparse
import logging
import socket
from queue import Queue

import numpy as np
import sounddevice as sd

logging.basicConfig(level=logging.INFO)


class SpeechCommandClient:
    """Speech Command Client.

    Args:
        host: str, default '127.0.0.1'
        port: int, default 51234
        samplerate: int, default 16000
        channels: int, default 1
        chunk_samples: int, default 8000
    """
    def __init__(
            self, 
            host: str = '127.0.0.1', 
            port: int = 51234, 
            samplerate: int = 16000, 
            channels: int = 1, 
            chunk_samples: int = 8000
        ):
        self.host = host
        self.port = port
        self.samplerate = samplerate
        self.channels = channels
        self.chunk_samples = chunk_samples
        self.queue = Queue()
        self.client_socket = None

    def audio_callback(self, indata: np.ndarray, frames: int, time: float, status: int):
        """Audio callback.

        Args:
            indata: np.ndarray, audio data
            frames: int, number of frames
            time: float, time
            status: int, status
        """
        self.queue.put(indata.copy())

    def connect(self) -> None:
        """Connect to the server."""
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((self.host, self.port))
        logging.info(f"Connected to server at {self.host}:{self.port}")

    def start_stream(self) -> None:
        """Start the audio stream."""
        with sd.InputStream(
                samplerate=self.samplerate, 
                channels=self.channels, 
                callback=self.audio_callback, 
                blocksize=self.chunk_samples
            ):
            logging.info("Audio stream started")
            try:
                while True:
                    audio_chunk = self.queue.get()
                    audio_bytes = (audio_chunk * 32768).astype(np.int16).tobytes()
                    self.client_socket.sendall(audio_bytes)

                    response = self.client_socket.recv(32).decode().strip()
                    logging.info(f"Received: {response}")

                    if response == '-0':
                        logging.info("No valid command detected")
                    else:
                        logging.info(f"Detected command: {response}")

            except KeyboardInterrupt:
                logging.info("Stopped by user")

    def run(self) -> None:
        """Run the client."""
        self.connect()
        self.start_stream()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Speech Command Client")
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Server host')
    parser.add_argument('--port', type=int, default=51234, help='Server port')
    parser.add_argument('--samplerate', type=int, default=16000, help='Audio sample rate')
    parser.add_argument('--channels', type=int, default=1, help='Number of audio channels')
    parser.add_argument('--chunk_samples', type=int, default=8000, help='Audio chunk samples')
    args = parser.parse_args()

    client = SpeechCommandClient(
        host=args.host,
        port=args.port,
        samplerate=args.samplerate,
        channels=args.channels,
        chunk_samples=args.chunk_samples
    )
    client.run()
