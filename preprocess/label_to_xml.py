from music21 import stream, note, meter, clef, tie

def duration_from_string(duration_str):
    if duration_str == 'quarter':
        return 1.0
    elif duration_str == 'eighth':
        return 0.5
    elif duration_str == 'half':
        return 2.0
    elif duration_str == 'whole':
        return 4.0
    else:
        # Handle other durations if needed
        return float(duration_str)

# Create a music21 Score object
score = stream.Score()
part = stream.Part()
measure = stream.Measure(number=1)

# Open and read the .txt file
with open('agnostic_output.txt', 'r') as file:
    lines = file.readlines()

# Parse each line and create corresponding music21 objects
for line in lines:
    line = line.strip()
    print(f"Processing line: {line}")

    parts = line.split()

    if line.startswith("Clef:"):
        clef_sign = parts[1]
        measure.append(clef.clefFromString(clef_sign))
    
    elif line.startswith("Time Signature:"):
        time_sig = parts[2]
        beats, beat_type = map(int, time_sig.split("/"))
        measure.append(meter.TimeSignature(f"{beats}/{beat_type}"))
    
    elif line.startswith("note-"):
        pitch = parts[0].split('-')[1]
        duration = duration_from_string(parts[1])
        n = note.Note(pitch)
        n.quarterLength = duration
        
        if len(parts) > 2 and parts[-1].startswith("tie"):
            if parts[-1] == "tie_start":
                n.tie = tie.Tie("start")
            elif parts[-1] == "tie_continue":
                n.tie = tie.Tie("continue")
        
        measure.append(n)
    
    elif line.startswith("rest"):
        duration = duration_from_string(parts[1])
        r = note.Rest()
        r.quarterLength = duration
        measure.append(r)
    
    else:
        print(f"Warning: Skipping unrecognized line: {line}")
    
    # Check if measure is full and create a new one if needed
    if measure.duration.quarterLength >= 4.0:  # Assuming 4/4 time signature
        part.append(measure)
        measure = stream.Measure(number=len(part.getElementsByClass('Measure')) + 1)

# Add any remaining notes in the last measure
if measure.notesAndRests:
    part.append(measure)

score.append(part)

# Write the score to a MusicXML file
output_file = 'output_from_agnostic.musicxml'
score.write('musicxml', output_file)

print(f"MusicXML file '{output_file}' has been successfully generated!")