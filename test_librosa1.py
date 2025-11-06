import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# --- Load audio ---
audio_file = r"C:\Users\caleb\Downloads\piano-chord-45626.mp3"
y, sr = librosa.load(audio_file, sr=None)

# --- Compute chroma features ---
chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr)

# --- Define chord templates ---
# Each template is a 12-note vector (C, C#, D, ..., B)
CHORD_TEMPLATES = {
    "C":    [1,0,0,0,0,0,1,0,0,1,0,0],       # C E G
    "Cmaj7":[1,0,0,0,0,0,1,0,0,1,0,1],       # C E G B
    "D":    [0,0,1,0,1,0,0,1,0,0,0,0],       # D F# A
    "Dm":   [0,0,1,0,1,0,0,0,1,0,0,0],       # D F A
    "E":    [0,0,0,1,0,1,0,0,1,0,0,0],       # E G# B
    "F":    [0,0,0,0,1,0,0,1,0,0,1,0],       # F A C
    "G":    [0,0,0,0,0,0,1,0,0,1,0,1],       # G B D
    "Am":   [1,0,0,0,1,0,0,1,0,0,0,0],       # A C E
    # Add more chords as needed
}

threshold = 0.3  # notes with energy > threshold are considered "on"

def match_chord(chroma_vec):
    notes_on = chroma_vec > threshold
    best_chord = "Unknown"
    best_match = -1
    for chord, template in CHORD_TEMPLATES.items():
        template = np.array(template) > 0
        match = np.sum(notes_on & template)
        if match > best_match:
            best_match = match
            best_chord = chord
    return best_chord

# --- Detect chords for each frame ---
chords = [match_chord(chroma[:, i]) for i in range(chroma.shape[1])]

# --- Smooth chord progression (merge consecutive duplicates) ---
smoothed_chords = []
smoothed_times = []
prev = None
for t, c in zip(times, chords):
    if c != prev:
        smoothed_chords.append(c)
        smoothed_times.append(t)
        prev = c

# --- Convert chords to numeric values for plotting ---
unique_chords = list(dict.fromkeys(smoothed_chords))  # preserves order
chord_to_num = {ch: i for i, ch in enumerate(unique_chords)}
y_vals = [chord_to_num[c] for c in smoothed_chords]

# --- Plot ---
plt.figure(figsize=(15, 4))
plt.scatter(smoothed_times, y_vals, c=y_vals, cmap="tab20", s=40)
plt.yticks(list(chord_to_num.values()), list(chord_to_num.keys()))
plt.xlabel("Time (s)")
plt.ylabel("Chord")
plt.title("Chord Progression")
plt.grid(True, alpha=0.3)
plt.show()
