import json

# Correct way to open and read a JSON file
with open(r"E:\Murphy_Model\final_files\json_data\main_json", "r", encoding="utf-8") as file:
    json_data = json.load(file)  # Load JSON content

# Extract all names if JSON is a list of dictionaries
if isinstance(json_data, list):
    disease_names = [entry["name"] for entry in json_data if "name" in entry]
else:  # If it's a single dictionary, extract just one name
    disease_names = [json_data["name"]] if "name" in json_data else []

# Print names
for name in disease_names:
    print(name)
