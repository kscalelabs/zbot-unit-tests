"""
Preparing the Environment:
```bash
scp -O test_07.py root@<KOS_IP_ADDRESS>:/root/alpine/root/
ssh root@<KOS_IP_ADDRESS>
chroot /root/alpine
cd /root
source venv/bin/activate
```

Running the Test:
```bash
python test_07.py
```

Retrieving the Recorded Audio:
```bash
scp -O root@<KOS_IP_ADDRESS>:/root/alpine/root/recorded_audio.wav .
```
"""
import argparse
import sounddevice as sd
import soundfile as sf

# Default parameters
DEFAULT_DURATION = 5  # Default recording duration in seconds

def main(duration):
    # File to store the recorded audio
    output_file = "recorded_audio.wav"

    # Device parameters
    record_device_id = 0      # Microphone device index
    playback_device_id = 1    # Speaker device index
    channels = [2]            # Use channel 2 of the microphone
    samplerate = 44100        # Sampling rate in Hz

    try:
        # Step 1: Record audio
        print(f"Recording for {duration} seconds...")
        audio_data = sd.rec(
            int(duration * samplerate), 
            samplerate=samplerate, 
            channels=len(channels), 
            device=record_device_id, 
            dtype='float32', 
            mapping=channels)
        sd.wait()  # Wait until recording is finished
        print("Recording finished.")

        # Save the recorded audio to a WAV file
        sf.write(output_file, audio_data, samplerate)
        print(f"Audio saved to {output_file}")

        # Step 2: Play back the recorded audio
        print(f"Playing back the recorded audio...")
        data, samplerate = sf.read(output_file)
        sd.play(data, samplerate=samplerate, device=playback_device_id)
        sd.wait()  # Wait for playback to complete
        print("Playback finished.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Record and play back audio with adjustable duration."
    )
    # Add argument for duration
    parser.add_argument(
        "--duration",
        type=int,
        default=DEFAULT_DURATION,
        help="Recording duration in seconds (default: 5).",
    )
    args = parser.parse_args()

    # Run the main function
    main(args.duration)