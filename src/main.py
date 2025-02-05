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
    # utils.evaluate_model_on_test_queries(inference)

    # # Example query
    query = [
        "how to make coffee",
        " what is the capital of france",
        "best programming language",
        "covid symptoms",
        "chocolate cake recipe",
    ]

    for q in query:
        print(f"Query: {q}")
        neighbours, scores = inference.kNN(q)
        for neighbour, score in zip(neighbours, scores):
            print(f"{neighbour}, Score: {score}")
        print("--------------------------------")


if __name__ == "__main__":
    main()
