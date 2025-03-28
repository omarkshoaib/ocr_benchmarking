import csv
import json
import os
import re

def create_json_split(csv_path, image_base_dir, output_json_path):
    """
    Reads a CSV file, processes image-transcription pairs, and writes to a JSON file.

    Args:
        csv_path (str): Path to the input CSV file.
        image_base_dir (str): Base directory where images for this split are located.
        output_json_path (str): Path to the output JSON file.
    """
    data = []
    skipped_header = False
    malformed_lines = 0

    try:
        with open(csv_path, 'r', encoding='utf-8', errors='ignore') as infile:
            for i, line in enumerate(infile):
                line = line.strip() # Remove leading/trailing whitespace including newline
                if not line:
                    continue # Skip empty lines

                # Skip header line (assuming it's the first line and looks like the numeric string)
                if i == 0 and line.isdigit() and len(line) > 50:
                    skipped_header = True
                    continue

                image_filename_base = None
                transcription = None
                original_filename = None

                # Find split point based on extension
                if '.tif' in line.lower():
                    try:
                        parts = line.split('.tif', 1)
                        original_filename = parts[0] + '.tif'
                        image_filename_base = parts[0]
                        transcription = parts[1]
                    except IndexError:
                        pass # Handle cases where split might fail unexpectedly
                elif '.jpg' in line.lower():
                     try:
                        parts = line.split('.jpg', 1)
                        original_filename = parts[0] + '.jpg'
                        image_filename_base = parts[0]
                        transcription = parts[1]
                     except IndexError:
                        pass # Handle cases where split might fail unexpectedly

                if image_filename_base and transcription is not None: # Check transcription is not None explicitly
                    # Ensure the filename uses .jpg and construct the full path
                    image_filename_jpg = f"{image_filename_base}.jpg"
                    image_path = os.path.join(image_base_dir, image_filename_jpg)
                    # Use relative path from project root
                    relative_image_path = os.path.relpath(image_path, start='.')

                    # Basic cleaning of transcription (remove potential leading/trailing whitespace)
                    transcription = transcription.strip()

                    if not transcription:
                         malformed_lines += 1
                         # print(f"Warning: Empty transcription for image {image_filename_jpg} in {csv_path}")
                         continue


                    data.append({
                        "image": relative_image_path.replace('\\', '/'), # Ensure forward slashes
                        "prompt": "Transcribe the text in this image.",
                        "transcription": transcription
                    })
                else:
                    malformed_lines += 1
                    # print(f"Warning: Could not parse line in {csv_path}: {line[:100]}...")


    except Exception as e:
        print(f"Error processing file {csv_path}: {e}")
        return # Stop processing this file on error

    if malformed_lines > 0:
        print(f"Warning: Skipped {malformed_lines} malformed or unparseable lines in {csv_path}.")

    if not data:
        print(f"Warning: No data extracted from {csv_path}. Check file format and paths.")
        return

    try:
        with open(output_json_path, 'w', encoding='utf-8') as outfile:
            json.dump(data, outfile, ensure_ascii=False, indent=2)
        print(f"Successfully created {output_json_path} with {len(data)} entries.")
    except Exception as e:
        print(f"Error writing JSON file {output_json_path}: {e}")


# Define paths
train_csv = 'khatt_dataset/Train.csv'
val_csv = 'khatt_dataset/Validation.csv'
train_img_dir = 'khatt_dataset/Train_deskewed/Train_deskewed'
val_img_dir = 'khatt_dataset/Validate_deskewed/Validate_deskewed' # Assuming this path based on Train
train_json = 'train.json'
val_json = 'val.json'

# Process Training Data
print("Processing Training Data...")
create_json_split(train_csv, train_img_dir, train_json)

# Process Validation Data
print("\nProcessing Validation Data...")
create_json_split(val_csv, val_img_dir, val_json)

print("\nDataset preparation script finished.")
