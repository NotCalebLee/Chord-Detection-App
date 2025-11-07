from chord_extractor import Chordino
import matplotlib.pyplot as plt

# Create the extractor
extractor = Chordino()

# Replace with your audio file path
audio_file = "C:/Users/caleb/Downloads/piano-chord-45626.mp3"

# Extract chords
chords = extractor.extract(audio_file)

# Print chords
for time, chord in chords:
    print(f"{time:.2f}s: {chord}")

# Plot chord progression
times = [t for t, c in chords]
labels = [c for t, c in chords]

plt.figure(figsize=(12, 3))
plt.scatter(times, labels)
plt.xlabel("Time (s)")
plt.ylabel("Chord")
plt.title("Chord Progression")
plt.show()
