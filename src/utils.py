"""
Utility functions for converting orders to and from JSON.
"""
import json
from classes import PizzaOrder, DrinkOrder 
from datasets import load_dataset, DatasetDict, Dataset
#from preprocess import *
import re
import ast


def order_to_json(order):
    """
    Convert a pizza or drink order to a JSON string.

    Args:
        order (PizzaOrder or DrinkOrder): The order to convert.

    Returns:
        str: The JSON string representing the order.
    """
    return json.dumps(order, default=lambda o: o.__dict__, indent=4)

def order_from_json(json_str):
    """
    Convert a JSON string to a pizza or drink order.

    Args:
        json_str (str): The JSON string to convert.

    Returns:
        PizzaOrder or DrinkOrder: The order object.
    """
    order_dict = json.loads(json_str)
    if 'size' in order_dict:
        return PizzaOrder(**order_dict)
    else:
        return DrinkOrder(**order_dict)

def read_test_cases(file_path):
    """
    Read test cases from a JSON file.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        list: A list of test cases.
    """
    with open(file_path, 'r') as file:
        contents = file.read()
        return contents
    
def read_labels(file_path):
    labels = []
    with open(file_path, 'r') as file:
        labels = file.read()
    labels = labels[:-1]

    label_list = labels.split("\n")
    return label_list

def read_unique_labels(file_path):
    labels = []
    with open(file_path, 'r') as file:
        labels = file.read().splitlines()
    return labels

def extract_sentences():

    # Extract src data
    source_data = []
    top_data = []
    top_decoupled_data = []

    # Read source.txt
    with open('../dataset/source.txt', 'r') as file:
        source_data = ast.literal_eval(file.read())

    # Read top.txt
    with open('../dataset/top.txt', 'r') as file:
        top_data = ast.literal_eval(file.read())

    # Read top_decoupled.txt
    with open('../dataset/top_decoupled.txt', 'r') as file:
        top_decoupled_data = ast.literal_eval(file.read())

    return source_data,top_data,top_decoupled_data

def convert_json_txt(file_path, output_file_path):
    top_data=[]
    with open(file_path, 'r') as file:
        lines=file.readlines()
        for i,line in enumerate(lines):
            print(line)
            regex = r'(?<="train\.SRC": ").+(?=", "train\.EXR")'
            exr_value = re.findall(regex,line)
            top_data.append(exr_value[0])

    # Define the output file path for the JSON file

    # Save the extracted data into a JSON file
    with open(output_file_path, 'w') as json_file:
        json.dump(top_data, json_file, indent=4)

    print("Data has been saved to", output_file_path)

    print(top_data[0])

def create_labels_file(file_path,output_file):
    try:
        data = load_dataset('json', data_files=file_path)
    except Exception as e:
        raise ValueError(f"Failed to load dataset from {file_path}: {e}")
    
    train_SRC_size = len(data['train']['train.SRC'])
    result = []
    tags = []
    longest_sentence = 0
    train_SRC_labels = []
    unique_words = set()
    with open('input_labels.txt', 'w') as f:
        for i in range(train_SRC_size):
            train_SRC = data['train']['train.SRC'][i]
            train_TOP_DECOUPLED = data['train']['train.TOP-DECOUPLED'][i]
            longest_sentence = max(len(train_SRC.split()), longest_sentence)    
            unique_words.update(train_SRC.split())            

            result.append(parse_tc(train_SRC,train_TOP_DECOUPLED))

            tags.append(generate_bio_tags(result[i]['sentence'], result[i]['entities']))

            for word, tag in tags[i]:
                train_SRC_labels.append(tag)
                f.write(f"{tag} ")
            f.write("\n")
            
if __name__ == '__main__':
    # Test read_test_cases
    #clearead_test_cases('./try.json')
    path_file = '../dataset/input_labels.txt'
    # output_file = 'input_labels2.txt'
    # create_labels_file(path_file, output_file)
    with open(path_file, 'r',encoding='utf-8') as file:
        source_data = file.read().split()

    set_source_data = set(source_data)
    set_source_data.remove('0')
    
    with open("unique_labels.txt", 'w', encoding='utf-8') as file:
        for label in set_source_data:
            file.write(f"{label}\n")

    