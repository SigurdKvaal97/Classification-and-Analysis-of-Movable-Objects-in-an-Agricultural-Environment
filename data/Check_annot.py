import json

def validate_json(file_path):
    with open(file_path, 'r') as file:
        try:
            json.load(file)
            print("JSON is valid")
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError: {e}")

validate_json('gaze_data_annot.json')