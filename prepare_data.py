import json

# Function to retrieve data from a JSON file with specified encoding
def retrieve_data_from_json(file_path, encoding='utf-8'):
    data = []
    try:
        with open(file_path, 'r', encoding=encoding) as file:
            json_data = json.load(file)
            for doc in json_data:
                content = doc.get("content", "")
                data.append(content)
    except UnicodeDecodeError as e:
        print(f"Encoding error reading JSON file: {e}")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON file: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    return data

# File path to your uploaded JSON file
json_file_path = 'test.json'

# Retrieve data from the JSON file using 'utf-8-sig' encoding
data = retrieve_data_from_json(json_file_path, encoding='utf-8-sig')

# If the previous attempt fails, try with 'latin-1' encoding
if not data:
    data = retrieve_data_from_json(json_file_path, encoding='latin-1')

# Get the first 100 documents
data = data[:5]

print(f"Total number of documents retrieved: {len(data)}")