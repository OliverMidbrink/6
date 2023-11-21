import json
import random

def get_targets_iPPIs(file_path):
    with open(file_path, "r") as file:
        return json.load(file)["tuples"]

def main():
    tuples = get_targets_iPPIs("MULTISELS/OSK_upreg_2_neighbors_chatGPT3_turpo_1106.json")
    
    for x in tuples:

if __name__ == "__main__":
    main()