import json

image_ids = [
    '1-079.jpg',
    '10-088.jpg',
    '11-089.jpg',
    '12-090.jpg',
]

with open("annotations/instances_default.json", 'r') as f:
    data = json.load(f)

ids = [image['id'] for image in data['images'] if image['file_name'] in image_ids]

annotations = [
    ann for ann in data['annotations'] if ann['image_id'] in ids
]

data['annotations'] = annotations

with open("annotations/instances_default_clean.json", 'w') as f:
    json.dump(data, f)
