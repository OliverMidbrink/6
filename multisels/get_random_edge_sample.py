import json
import random

def load_edge_list():
    edge_list = []
    with open("interactome/HuRI_uniprot_edge_list.json", "r") as file:
        edge_list = json.load(file)["uniprot_edges"]

    return edge_list

def get_random_edge_sample(n=1000):
    edge_list = load_edge_list()
    return random.sample(edge_list, n)


def main():
    sample = get_random_edge_sample(n=10)
    print(sample)

if __name__ == "__main__":
    main()