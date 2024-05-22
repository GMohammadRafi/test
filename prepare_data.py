import json

# Function to retrieve data from a JSON file
def retrieve_data_from_json(file_path):
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            json_data = json.load(file)
            for doc in json_data:
                data.append(
                    {
                        "content": doc['content'],
                        "_id": doc['_id'],
                        "tags": doc['tags']
                    })
    except Exception as e:
        print(f"Error reading JSON file: {e}")
    return data

# File path to your uploaded JSON file
json_file_path = 'test_a.json'

# Retrieve data from the JSON file
data = retrieve_data_from_json(json_file_path)

print(f"Total number of documents retrieved and inserted: {len(data)}")
