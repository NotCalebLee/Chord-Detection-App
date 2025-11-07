import librosa
import numpy as np
import matplotlib.pyplot as plt

# Load audio
audio_file = "C:/Users/caleb/Downloads/piano-chord-45626.mp3"
y, sr = librosa.load(audio_file, sr=None)

# Compute chroma features (pitch class profiles)
chroma = librosa.feature.chroma_cqt(y=y, sr=sr)

# Average chroma over short windows to get chord per frame
hop_length = 512
chroma_avg = np.mean(chroma, axis=1)

# Simple chord detection helper
CHORDS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def detect_chord(chroma_vector):
    # Pick the pitch class with highest energy
    index = np.argmax(chroma_vector)
    return CHORDS[index]

# Slide over frames
frame_length = 50  # number of chroma frames per step
times = []
labels = []

for i in range(0, chroma.shape[1], frame_length):
    frame = chroma[:, i:i+frame_length]
    chord = detect_chord(np.mean(frame, axis=1))
    times.append(i * hop_length / sr)
    labels.append(chord)

# Print chord progression
for t, c in zip(times, labels):
    print(f"{t:.2f}s: {c}")

# Plot
plt.figure(figsize=(12, 3))
plt.scatter(times, labels)
plt.xlabel("Time (s)")
plt.ylabel("Chord")
plt.title("Chord Progression (approx.)")
plt.show()
