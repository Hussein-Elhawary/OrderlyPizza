"""
Utility functions for converting orders to and from JSON.
"""
import json
from classes import PizzaOrder, DrinkOrder 
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


if __name__ == '__main__':
    # Test read_test_cases
    read_test_cases('./try.json')