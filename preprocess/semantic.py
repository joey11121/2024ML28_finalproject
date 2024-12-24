import os
import xml.etree.ElementTree as ET

def parse_xml_to_labels(xml_file, output_type="semantic"):
    """
    Parse an XML file into semantic or agnostic labels, handling mid-measure clef changes
    while preventing duplicate initial clef.
    
    Parameters:
        xml_file (str): Path to the XML file.
        output_type (str): Either 'semantic' or 'agnostic'.
    
    Returns:
        str: Formatted output labels.
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    labels = []
    start = True
    initial_clef = None

    # Process initial clef, keySignature, and timeSignature
    for measure in root.iter('measure'):
        if measure.get('number') == '1':
            # Handle initial clef and store it
            clef = measure.find(".//clef")
            if clef is not None:
                clef_sign = clef.find("sign").text if clef.find("sign") is not None else "G"
                clef_line = clef.find("line").text if clef.find("line") is not None else "2"
                initial_clef = f"clef-{clef_sign}{clef_line}"
                labels.append(initial_clef)
            
            # Handle initial key signature
            key = measure.find(".//key")
            if key is not None:
                major_keys = {
                    "-7": "Cb", "-6": "Gb", "-5": "Db", "-4": "Ab", "-3": "Eb", "-2": "Bb", "-1": "F",
                    "0": "C", "1": "G", "2": "D", "3": "A", "4": "E", "5": "B", "6": "F#", "7": "C#"
                }
                minor_keys = {
                    "-7": "Ab", "-6": "Eb", "-5": "Bb", "-4": "F", "-3": "C", "-2": "G", "-1": "D",
                    "0": "A", "1": "E", "2": "B", "3": "F#", "4": "C#", "5": "G#", "6": "D#", "7": "A#"
                }
                key_fifths = key.find("fifths").text if key.find("fifths") is not None else "0"
                key_mode = key.find("mode").text if key.find("mode") is not None else "major"

                if key_mode == "minor":
                    key_name = minor_keys[key_fifths] + "m" if key_fifths in minor_keys else "Am"
                else:
                    key_name = major_keys[key_fifths] + "M" if key_fifths in major_keys else "CM"
                
                labels.append(f"keySignature-{key_name}")

            # Handle initial time signature
            time = measure.find(".//time")
            if time is not None:
                beats = time.find("beats").text if time.find("beats") is not None else "4"
                beat_type = time.find("beat-type").text if time.find("beat-type") is not None else "4"
                labels.append(f"timeSignature-{beats}/{beat_type}")
            break

    # Process measures and their contents
    for measure in root.iter('measure'):
        if not start:
            labels.append("barline")
        start = False

        # Process each element in the measure
        for element in measure:
            if element.tag == 'attributes':
                # Handle mid-measure clef changes, skip if it's the same as initial clef
                clef = element.find('clef')
                if clef is not None:
                    clef_sign = clef.find("sign").text if clef.find("sign") is not None else "G"
                    clef_line = clef.find("line").text if clef.find("line") is not None else "2"
                    new_clef = f"clef-{clef_sign}{clef_line}"
                    # Only add if it's different from the initial clef and not in first measure
                    if new_clef != initial_clef or measure.get('number') != '1':
                        labels.append(new_clef)

            elif element.tag == "note":
                pitch = element.find("pitch")
                note_type = element.find("type")
                duration = note_type.text if note_type is not None else "eighth"
                #if note_type.text == '16th': duration = 'sixteenth'
                 # Handle special duration cases
                if duration == '16th': 
                    duration = 'sixteenth'
                elif duration == '32nd':
                    duration = 'thirty_second'
                elif duration == '64th':
                    duration = 'sixty_fourth'

                has_dot = element.find("dot") is not None
                dot_suffix = "." if has_dot else ""

                if pitch is not None:
                    step = pitch.find("step").text
                    alter_value = pitch.find("alter").text if pitch.find("alter") is not None else "0"
                    if alter_value == "1":
                        alter = "#"
                    elif alter_value == "-1":
                        alter = "b"
                    else:
                        alter = ""
                    octave = pitch.find("octave").text
                    labels.append(f"note-{step}{alter}{octave}_{duration}{dot_suffix}")
                else:
                    labels.append(f"rest-{duration}{dot_suffix}")
                
                # Handle ties
                tie_start = any(tie.attrib.get("type") == "start" for tie in element.findall("tie"))
                if tie_start:
                    labels.append("tie")

    return "    ".join(labels)

def process_directory(input_dir, output_dir, output_type="semantic"):
    """
    Process all XML files in a directory and output labels to .txt files.
    
    Parameters:
        input_dir (str): Path to the directory containing XML files.
        output_dir (str): Path to the directory where .txt files will be saved.
        output_type (str): Either 'semantic' or 'agnostic'.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".xml"):
            input_path = os.path.join(input_dir, file_name)
            output_path = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}.semantic")
            
            # Parse XML and generate labels
            labels = parse_xml_to_labels(input_path, output_type=output_type)
            
            # Write labels to the output .txt file
            with open(output_path, "w") as output_file:
                output_file.write(labels)
            print(f"Processed {file_name} -> {output_path}")

# Example usage
input_directory = "./test_primus"
output_directory = "./test_output"
process_directory(input_directory, output_directory, output_type="semantic")

