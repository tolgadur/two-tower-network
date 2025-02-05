import utils
import inference


def main():
    # tokenizer, _ = utils.load_tokenizer_and_embeddings()
    # tower_one = utils.load_tower_one()
    # tower_two = utils.load_tower_two()

    # kNNInference = inference.TwoTowerInference(tower_one, tower_two, tokenizer)
    # kNNInference.add_documents_from_file()

    # neighbours, scores = kNNInference.kNN("what is rba")
    # for neighbour, score in zip(neighbours, scores):
    #     print(f"Neighbour: {neighbour}, Score: {score}")
    #     print("--------------------------------")

    utils.train_two_tower()


if __name__ == "__main__":
    main()
