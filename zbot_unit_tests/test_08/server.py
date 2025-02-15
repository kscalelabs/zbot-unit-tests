"""Demo for the voice2action.

Run server locally.
"""
import argparse
import logging
import socket

import numpy as np
import onnxruntime as ort
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi

logging.basicConfig(level=logging.INFO)

_LABELS = {
    0: "unknown",
    1: "zbot", 
    2: "yes", 
    3: "no", 
    4: "left", 
    5: "right", 
    6: "on", 
    7: "stop", 
    8: "go", 
    9: "unknown", 
    10: "unknown",
}


class SpeechCommandServer:
    """
    Initialize the SpeechCommandServer.

    Args:
        host: str, default '0.0.0.0'
        port: int, default 51234
    """
    def __init__(
            self, 
            host: str = '0.0.0.0', 
            port: int = 51234, 
            chunk_samples: int = 8000, 
            model_path: str = 'model.onnx'
        ):
        self.host = host
        self.port = port
        self.chunk_samples = chunk_samples
        self.ort_session = ort.InferenceSession(model_path)
        self.in_cache = np.zeros((1, 64, 244), dtype=np.float32)
        self.labels = _LABELS
        self.server_socket = None
        self.conn = None
        self.addr = None
        self.save_audio = bytearray()

    def compute_mfcc(
            self, 
            waveform: torch.Tensor, 
            sample_rate: int = 16000, 
            num_ceps: int = 80, 
            num_mel_bins: int = 80
        ) -> np.ndarray:
        """
        Compute MFCC features from the waveform.

        Args:
            waveform: torch.Tensor, shape (1, T)
            sample_rate: int, default 16000
            num_ceps: int, default 80
            num_mel_bins: int, default 80

        Returns:
            np.ndarray, shape (num_ceps, num_mel_bins)
        """
        # waveform is in -32768 to 32767 ranges
        return kaldi.mfcc(
            waveform,
            num_ceps=num_ceps,
            num_mel_bins=num_mel_bins,
            frame_length=25,
            frame_shift=10,
            dither=0.0,
            energy_floor=0.0,
            sample_frequency=sample_rate,
        ).numpy()

    def recv_exactly(self, n: int) -> bytearray:
        """
        Receive exactly n bytes from the connection.

        Args:
            n: int, number of bytes to receive
        """
        buffer = bytearray()
        while len(buffer) < n:
            packet = self.conn.recv(n - len(buffer))
            if not packet:
                raise ConnectionError("Client disconnected")
            buffer.extend(packet)
        return buffer

    def start(self) -> None:
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        logging.info(f"Server listening on {self.host}:{self.port}")

        self.conn, self.addr = self.server_socket.accept()
        logging.info(f"Connected by {self.addr}")

        try:
            while True:
                data = self.recv_exactly(self.chunk_samples * 2)
                self.save_audio.extend(data)
                audio_chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32)

                waveform = torch.from_numpy(audio_chunk).unsqueeze(0)
                feats = self.compute_mfcc(waveform, num_ceps=80, num_mel_bins=80)
                feats = np.expand_dims(feats, axis=0)

                logits, _ = self.ort_session.run(None, {"input": feats, "cache": self.in_cache})

                probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
                probs /= np.sum(probs, axis=-1, keepdims=True)
                probs = probs.squeeze()

                predicted_idx = np.argmax(logits, axis=-1)[0]
                predicted_label = self.labels.get(predicted_idx, "unknown")

                probs_str = ' '.join([f'{p:.4f}' for p in probs])
                logging.info(f"Predicted: {predicted_label}, Probs: [{probs_str}]")

                response = f"{predicted_label if predicted_label != 'unknown' else '-0'}".ljust(32)
                self.conn.sendall(response.encode())

        except Exception as e:
            logging.error(f"Error: {e}")

        finally:
            self.cleanup()

    def cleanup(self):
        if self.save_audio:
            audio_data = np.frombuffer(self.save_audio, dtype=np.int16).astype(np.float32)
            waveform = torch.from_numpy(audio_data).unsqueeze(0)
            torchaudio.save('recorded_audio.wav', waveform, 16000)
            logging.info("Audio saved to recorded_audio.wav")

        if self.conn:
            self.conn.close()
        if self.server_socket:
            self.server_socket.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Speech Command Server")
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Server host')
    parser.add_argument('--port', type=int, default=51234, help='Server port')
    parser.add_argument('--chunk_samples', type=int, default=8000, help='Chunk samples')
    parser.add_argument('--model_path', type=str, default='model.onnx', help='Path to the ONNX model')
    args = parser.parse_args()

    server = SpeechCommandServer(
        host=args.host,
        port=args.port,
        chunk_samples=args.chunk_samples,
        model_path=args.model_path
    )
    server.start()

