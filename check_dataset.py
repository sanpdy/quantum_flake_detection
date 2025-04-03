import json
import os

# Paths to your annotation file and the train folder
annotation_file = "/home/sankalp/quant_flakes/quantumml/DL_2DMaterials/BN/annotations/instances_train2019.json"  # change if needed
train_folder = "/home/sankalp/quant_flakes/quantumml/DL_2DMaterials/BN/train2019"  # adjust path as necessary
output_file = "missing_files_BN.txt"

# Load the annotation file
with open(annotation_file, 'r') as f:
    data = json.load(f)

# Extract the filenames from the annotations (assuming they are stored in the "images" field)
annotated_files = [img["file_name"] for img in data.get("images", [])]

# Check which filenames are missing in the train folder
missing_files = []
for fname in annotated_files:
    # Construct the full path to the file in the train folder
    file_path = os.path.join(train_folder, fname)
    if not os.path.exists(file_path):
        missing_files.append(fname)

# Write out the missing filenames to the output file
with open(output_file, 'w') as f:
    for missing in missing_files:
        f.write(missing + "\n")

print(f"Checked {len(annotated_files)} files. {len(missing_files)} files are missing.")
print(f"Missing file list written to: {output_file}")
