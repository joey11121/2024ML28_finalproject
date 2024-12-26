import os
import subprocess
from music21 import *
from pathlib import Path
import logging
from midiutil import MIDIFile
import re
import glob
import pretty_midi

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_time_signature(semantic_content: str) -> str:
    """Extract time signature from semantic file content"""
    pattern = r'timeSignature-[^,\s]+'
    matches = re.findall(pattern, semantic_content)
    return matches[0] if matches else None

def add_time_signature(semantic_content: str, time_signature: str) -> str:
    """Ensure the correct time signature is present in the semantic content."""
    # Check if any timeSignature exists
    pattern = r'timeSignature-[^,\s]+'
    if re.search(pattern, semantic_content):
        # Replace existing timeSignature with the correct one
        semantic_content = re.sub(pattern, time_signature, semantic_content)
    else:
        # Add the correct timeSignature after keySignature
        key_signature_pattern = r'(keySignature-[^,\s]+)'
        if re.search(key_signature_pattern, semantic_content):
            semantic_content = re.sub(key_signature_pattern, r'\1\t' + time_signature, semantic_content)
        else:
            # Fallback if no keySignature is found (add at the start)
            semantic_content = time_signature + '\t' + semantic_content
    return semantic_content


def create_part_midi(midi_files: list, output_file: str, part_indices: list):
    """Create a MIDI file from selected staves"""
    try:
        combined_score = stream.Score()
        combined_part = stream.Part()
        
        # Sort and filter files for this part
        sorted_files = sorted(midi_files, key=lambda x: int(re.search(r'staff_(\d+)', x).group(1)))
        part_files = [f for i, f in enumerate(sorted_files, 1) if i in part_indices]
        
        # Combine all files for this part
        for midi_file in part_files:
            score = converter.parse(midi_file)
            for element in score.recurse():
                combined_part.append(element)
                
        combined_score.append(combined_part)
        combined_score.write('midi', fp=output_file)
        return True
        
    except Exception as e:
        logger.error(f"Error creating part MIDI: {e}")
        return False

def combine_part_midis(part_files: list, output_file: str):
    """Combine separate part MIDI files into final score"""
    try:
        combined_score = stream.Score()
        
        # Add each part
        for part_file in part_files:
            part = converter.parse(part_file)
            combined_score.append(part)
            
        combined_score.write('midi', fp=output_file)
        return True
        
    except Exception as e:
        logger.error(f"Error combining parts: {e}")
        return False

def convert_semantic_to_mei(semantic_file: str, output_mei: str) -> bool:
    """Convert .semantic file to .mei using semantic_conversor.sh"""
    try:
        result = subprocess.run(['./semantic_conversor.sh', semantic_file, output_mei],
                              capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Error converting {semantic_file}: {result.stderr}")
            return False
        return True
    except Exception as e:
        logger.error(f"Exception during conversion of {semantic_file}: {e}")
        return False

def convert_mei_to_midi(mei_file: str, midi_file: str, bpm: float):
    """Convert .mei file to .midi using music21"""
    try:
        # Parse the MEI file
        mei_score = converter.parse(mei_file)
        
        # Create and add tempo mark
        tempo_mark = tempo.MetronomeMark(number=bpm)
        mei_score.insert(0, tempo_mark)
        
        # Write to MIDI
        mei_score.write('midi', fp=midi_file)
        return True
    except Exception as e:
        logger.error(f"Error converting {mei_file} to MIDI: {e}")
        return False



def final_midis(output_file: str):
    """
    Combine multiple part MIDI files (part1.mid, part2.mid, etc.) into a single MIDI 
    where all parts play simultaneously.
    
    Args:
        output_file: Output MIDI file path
    """
    try:
        # Create a new PrettyMIDI object
        combined_midi = pretty_midi.PrettyMIDI()

        # Get all part files and convert paths to strings
        part_files = sorted(glob.glob("./midi_results/part*.mid"))
        if not part_files:
            logging.error("No 'part*.mid' files found!")
            return False
        
        logging.info(f"Found {len(part_files)} part files to combine.")

        # Add each part to the combined MIDI
        for i, part_file in enumerate(part_files):
            # Load the MIDI part
            try:
                part = pretty_midi.PrettyMIDI(str(part_file))  # Convert path to string
                
                # Add all instruments from this part
                for instrument in part.instruments:
                    # Assign a unique program number and channel to avoid conflicts
                    instrument.program = i
                    instrument.channel = i % 16
                    combined_midi.instruments.append(instrument)
                
                logging.info(f"Added part from {part_file} to combined MIDI.")
            
            except Exception as e:
                logging.error(f"Error processing {part_file}: {e}")
                continue

        # Write the combined MIDI file - convert output path to string
        combined_midi.write(str(output_file))
        logging.info(f"Successfully created combined MIDI file: {output_file}")
        return True

    except Exception as e:
        logging.error(f"Error combining part MIDIs: {e}")
        return False
def process_semantic_files(input_dir: str, output_dir: str, bpm: float, num_parts: int):
    """Process all .semantic files in input directory"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all .semantic files
    semantic_files = list(Path(input_dir).glob('*.semantic'))
    total_files = len(semantic_files)
    logger.info(f"Found {total_files} .semantic files to process")
    
    # Sort semantic files by staff number
    sorted_semantic_files = sorted(semantic_files, 
                                 key=lambda x: int(re.search(r'staff_(\d+)', x.stem).group(1)))
    
    # Extract time signature from first file
    with open(sorted_semantic_files[0], 'r') as f:
        first_content = f.read()
        time_signature = extract_time_signature(first_content)
        
    
    if time_signature:
        # Add time signature to all files except first
        for semantic_file in sorted_semantic_files[1:]:
           # print(time_signature)
            with open(semantic_file, 'r') as f:
                content = f.read()
            modified_content = add_time_signature(content, time_signature)
            with open(semantic_file, 'w') as f:
                f.write(modified_content)
    
    # Process files for each part
    for part_num in range(1, num_parts + 1):
        part_semantic_files = [f for i, f in enumerate(sorted_semantic_files, 1) 
                             if i % num_parts == part_num % num_parts]
        
        # Process each file in the part
        part_midi_files = []
        for semantic_file in part_semantic_files:
            base_name = semantic_file.stem
            mei_file = Path(output_dir) / f"{base_name}.mei"
            midi_file = Path(output_dir) / f"{base_name}.mid"
            
            if convert_semantic_to_mei(str(semantic_file), str(mei_file)):
                if convert_mei_to_midi(str(mei_file), str(midi_file), bpm):
                    part_midi_files.append(str(midi_file))
                    logger.info(f"Successfully created {midi_file}")
                else:
                    logger.error(f"Failed to create MIDI for {base_name}")
            else:
                logger.error(f"Failed to create MEI for {base_name}")
            
            if mei_file.exists():
                mei_file.unlink()
        
        # Create combined part MIDI
        if part_midi_files:
            part_file = Path(output_dir) / f"part{part_num}.mid"
            combine_part_midis(part_midi_files, str(part_file))
            logger.info(f"Created part {part_num}")
    
    # Combine all parts into final score
    part_files = [Path(output_dir) / f"part{i}.mid" for i in range(1, num_parts + 1)]
    if all(f.exists() for f in part_files):
        combined_midi = Path(output_dir) / "combined_score.mid"
        final_midis(combined_midi)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert semantic files to MIDI')
    parser.add_argument('--input_dir', default='stave_semantic_results',
                        help='Directory containing .semantic files')
    parser.add_argument('--output_dir', default='midi_results',
                        help='Directory to save MEI and MIDI files')
    parser.add_argument('--bpm', type=float, default=120.0,
                        help='Tempo in beats per minute')
    parser.add_argument('--num_parts', type=int, default=1,
                        help='Number of voice parts in the score')
    
    args = parser.parse_args()
    
    logger.info(f"Processing files from {args.input_dir}")
    logger.info(f"Saving results to {args.output_dir}")
    logger.info(f"Using tempo: {args.bpm} BPM")
    logger.info(f"Number of parts: {args.num_parts}")
    
    process_semantic_files(args.input_dir, args.output_dir, args.bpm, args.num_parts)
    
    logger.info("Processing complete!")

if __name__ == "__main__":
    main()

