import json

# Paths to the original and output annotation files
input_annotation_file = "/home/sankalp/quant_flakes/quantumml/DL_2DMaterials/BN/annotations/instances_val2019.json"
output_annotation_file = "/home/sankalp/quant_flakes/quantumml/DL_2DMaterials/BN/annotations/instances_val2019_clean.json"

# Load the original annotation file
with open(input_annotation_file, 'r') as f:
    data = json.load(f)

# Create a mapping from old image IDs (string) to new im
# age IDs (integer)
old_to_new_id = {}
new_images = []
for idx, image in enumerate(data["images"]):
    new_id = idx + 1  # Starting IDs from 1
    old_to_new_id[image["id"]] = new_id
    # Update the image's id
    image["id"] = new_id
    new_images.append(image)
data["images"] = new_images

# Update the annotations to reflect the new image IDs
new_annotations = []
for ann in data["annotations"]:
    old_image_id = ann["image_id"]
    # Replace with the new image id using the mapping
    if old_image_id in old_to_new_id:
        ann["image_id"] = old_to_new_id[old_image_id]
        new_annotations.append(ann)
    else:
        print(f"Warning: Annotation found with unknown image_id {old_image_id}")
data["annotations"] = new_annotations

# Save the cleaned annotation file
with open(output_annotation_file, 'w') as f:
    json.dump(data, f)

print(f"Cleaned annotation file saved as {output_annotation_file}")
