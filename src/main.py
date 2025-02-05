import utils
import inference


def main():
    tokenizer, _ = utils.load_tokenizer_and_embeddings()
    tower_one = utils.load_tower_one()
    tower_two = utils.load_tower_two()

    kNNInference = inference.TwoTowerInference(tower_one, tower_two, tokenizer)
    kNNInference.encode_documents_by_filename()

    neighbours, scores = kNNInference.kNN("what is rba")
    print(neighbours)
    print(scores)


if __name__ == "__main__":
    main()
