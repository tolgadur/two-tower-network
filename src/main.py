import utils
from inference import TwoTowerInference


def main():

    # utils.train_two_tower()
    # Load models
    tower_one = utils.load_tower_one()
    tower_two = utils.load_tower_two()

    # Initialize inference
    inference = TwoTowerInference(tower_one, tower_two)
    inference.add_documents_from_file()

    # Example query
    query = "what is rba"
    neighbours, scores = inference.kNN(query)
    for neighbour, score in zip(neighbours, scores):
        print(f"Neighbour: {neighbour}, Score: {score}")
        print("--------------------------------")


if __name__ == "__main__":
    main()
