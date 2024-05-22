import json
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
import re
from prepare_data import data

# Configure logging
logging.basicConfig(filename='generation_log.txt', level=logging.INFO, format='%(asctime)s - %(message)s')

# Load the model and tokenizer
model_name = "rhysjones/phi-2-orange-v2"  # 160.97 # 15.00
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Set the device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

data_tmp = []

for index, item in enumerate(data):
    input_text = item['content']

    # Encode the input text and move it to the appropriate device
    input_ids = tokenizer.encode(input_text + " For a given paragraph craft a title. give me the list of key word",
                                 return_tensors="pt").to(device)

    # Start the timer
    start_time = time.time()

    # Generate the response
    output_ids = model.generate(input_ids, max_length=512, num_return_sequences=1)

    # Decode the generated text
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # End the timer
    end_time = time.time()

    title_match = re.search(r'Title: "(.*?)"', output_text)
    keywords_match = re.search(r'Key words: (.*)\n', output_text)

    if title_match and keywords_match:
        generated_title = title_match.group(1)
        generated_keywords = keywords_match.group(1).split(", ")

        # Add title to the item
        item['title'] = generated_title

        # Convert existing tags to lowercase and remove duplicates
        existing_tags = set(tag.lower() for tag in item['tags'])

        # Add new keywords to tags, converting to lowercase (except for "AI")
        for keyword in generated_keywords:
            if keyword.lower() != "ai":
                existing_tags.add(keyword.lower())
            else:
                existing_tags.add("AI")

        # Update the item's tags
        item['tags'] = list(existing_tags)
    data_tmp.append(item)

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    # Log the index, output, and the elapsed time
    log_message = f"=====================\nID: {item['_id']}\nTime taken {index} : {elapsed_time:.2f} seconds"
    logging.info(log_message)

# Write the modified data to test_v2.json
with open('test_v2.json', 'w') as outfile:
    json.dump(data_tmp, outfile, indent=4)
