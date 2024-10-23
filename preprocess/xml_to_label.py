from music21 import converter, note, clef, meter, stream, tie

def get_staff_position(n):
    """
    Get staff position for a note object.
    Returns Lx, where x is the line or space on the staff.
    """
    if isinstance(n, note.Note):
        return f"L{n.pitch.diatonicNoteNum % 7 + 1}"
    return ""

def duration_to_string(duration):
    if duration == 1.0:
        return "quarter"
    elif duration == 0.5:
        return "eighth"
    elif duration == 2.0:
        return "half"
    elif duration == 4.0:
        return "whole"
    else:
        return f"{duration} quarter"

def get_tie_info(note_obj):
    if note_obj.tie is not None:
        if note_obj.tie.type == 'start':
            return "tie_start"
        elif note_obj.tie.type in ['continue', 'stop']:
            return "tie_continue"
    return ""

# Load the MusicXML file
score = converter.parse('output_music_test.musicxml')

# Open a file to write the agnostic representation
with open('agnostic_output.txt', 'w') as f:
    # Iterate through all parts in the score
    for part in score.parts:
        # Access different elements of the part
        for measure in part.getElementsByClass(stream.Measure):
            for element in measure.notesAndRests:
                if isinstance(element, clef.Clef):
                    f.write(f"Clef: {element.sign}\n")
                elif isinstance(element, meter.TimeSignature):
                    f.write(f"Time Signature: {element.ratioString}\n")
                elif isinstance(element, note.Note):
                    duration = duration_to_string(element.quarterLength)
                    position = get_staff_position(element)
                    tie_info = get_tie_info(element)
                    note_info = f"note-{element.nameWithOctave} {duration}"
                    if tie_info:
                        note_info += f" {tie_info}"
                    f.write(f"{note_info}\n")
                elif isinstance(element, note.Rest):
                    duration = duration_to_string(element.quarterLength)
                    f.write(f"rest {duration}\n")

print("Agnostic representation with tie information successfully written to agnostic_output.txt")