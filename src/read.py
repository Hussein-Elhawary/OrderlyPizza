import json
import re

#extract src data
top_data=[]
file_path =  '../dataset/PIZZA_train.json'
with open(file_path, 'r') as file:
    lines=file.readlines()
    for i,line in enumerate(lines):
        
        regex = r'(?<="train.TOP-DECOUPLED": ").+(?="})'
        exr_value = re.findall(regex,line)
        top_data.append(exr_value[0])

# Define the output file path for the JSON file
output_file_path = 'top_decoupled.txt'  # replace with desired output file path

# Save the extracted data into a JSON file
with open(output_file_path, 'w') as json_file:
    json.dump(top_data, json_file, indent=4)

print("Data has been saved to", output_file_path)

print(top_data[0])