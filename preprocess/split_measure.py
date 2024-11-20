from music21 import *
import os
import math
import copy

def split_musicxml_by_measure(input_file, output_dir, measures_per_file=2):
    """
    Split a MusicXML file into separate files with specified number of measures per file,
    preserving clef and key signature information across splits.
    
    Parameters:
    input_file (str): Path to the input MusicXML file
    output_dir (str): Directory where the split measures will be saved
    measures_per_file (int): Number of measures to include in each output file
    
    Returns:
    list: List of paths to the generated files
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the score
    score = converter.parse(input_file)

    # Get the file name without extension
    base_name = os.path.splitext(os.path.basename(input_file))[0]

    generated_files = []

    # Get all measures from the first part to determine total measure count
    measures = list(score.parts[0].getElementsByClass('Measure'))
    total_measures = len(measures)

    # Calculate number of output files needed
    num_files = math.ceil(total_measures / measures_per_file)

    def get_active_clef_and_key(part, measure_index):
        """Helper function to get the active clef and key signature at a given measure index."""
        clefs = []
        keys = []
        for m in list(part.getElementsByClass('Measure'))[:measure_index + 1]:
            clefs.extend(m.getElementsByClass('Clef'))
            keys.extend(m.getElementsByClass('KeySignature'))
        active_clef = clefs[-1] if clefs else None
        active_key = keys[-1] if keys else None
        return active_clef, active_key

    # Iterate through groups of measures
    for file_num in range(num_files):
        # Calculate start and end indices for this group
        start_idx = file_num * measures_per_file
        end_idx = min((file_num + 1) * measures_per_file, total_measures)

        # Create a new score for this group of measures
        new_score = stream.Score()

        # For each part in the original score
        for part in score.parts:
            # Create a new part
            new_part = stream.Part()

            # Get measures for this part
            part_measures = list(part.getElementsByClass('Measure'))

            # If this is not the first file, get the active clef and key from the previous section
            if file_num > 0:
                active_clef, active_key = get_active_clef_and_key(part, start_idx - 1)
                if active_clef:
                    # Create a copy of the clef
                    clef_copy = copy.deepcopy(active_clef)
                    # Insert the clef at the start of the new part
                    new_part.insert(0, clef_copy)
                if active_key:
                    # Create a copy of the key signature
                    key_copy = copy.deepcopy(active_key)
                    # Insert the key signature at the start of the new part
                    new_part.insert(0, key_copy)

            # Add the measures for this group
            for i in range(start_idx, end_idx):
                if i < len(part_measures):
                    # Create a deep copy of the measure
                    measure = copy.deepcopy(part_measures[i])

                    # If this is not the first measure of the first file,
                    # we might need to adjust the measure number
                    if i > 0:
                        measure.number = i + 1

                    new_part.append(measure)

            # Add the part to the new score if it has measures
            if len(new_part) > 0:
                new_score.append(new_part)

        # Generate output filename
        start_measure = start_idx + 1
        end_measure = end_idx
        output_file = os.path.join(
            output_dir, f"{base_name}_measures_{start_measure}-{end_measure}.xml"
        )

        # Write the measures to a new file
        new_score.write('musicxml', output_file)

        generated_files.append(output_file)

    return generated_files

def process_all_files_in_folder(input_folder, output_folder, measures_per_file=3):
    """
    Process all .mxl files in the input folder, splitting each by measure.
    
    Parameters:
    input_folder (str): Path to the folder containing .mxl files
    output_folder (str): Path to the folder to save split files
    measures_per_file (int): Number of measures to include in each split file
    """
    # Get all .mxl files in the input folder
    mxl_files = [f for f in os.listdir(input_folder) if f.endswith('.mxl')]

    if not mxl_files:
        print("No .mxl files found in the input folder.")
        return

    for mxl_file in mxl_files:
        input_file_path = os.path.join(input_folder, mxl_file)
        print(f"Processing {mxl_file}...")
        try:
            generated_files = split_musicxml_by_measure(
                input_file_path, output_folder, measures_per_file
            )
            print(f"Successfully split {mxl_file} into {len(generated_files)} files:")
            for file in generated_files:
                print(f"- {file}")
        except Exception as e:
            print(f"Error processing {mxl_file}: {str(e)}")

def main():
    # Example usage
    input_folder = "./input_files"  # Folder with .mxl files
    output_folder = "./split_output"  # Folder to save split files
    measures_per_file = 1  # Number of measures per split file

    process_all_files_in_folder(input_folder, output_folder, measures_per_file)

if __name__ == "__main__":
    main()
