import music21

# Load a built-in Bach piece that comes with music21
score = music21.corpus.parse('bach/bwv66.6')

print("Piece loaded successfully!")
print(f"Number of parts: {len(score.parts)}")

# Print each part name and first few notes
for i, part in enumerate(score.parts):
    notes = part.flat.notes[:5]
    print(f"\nPart {i+1}: {part.partName}")
    for note in notes:
        print(f"  Note: {note.nameWithOctave} | Frequency: {round(note.pitch.frequency, 1)} Hz")