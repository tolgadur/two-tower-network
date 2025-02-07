from inference import Inference
import pandas as pd


def test_accuracy(inference: Inference, k: int = 1):
    print(f"\nTesting model accuracy with k={k}...")

    # Load test triplets
    test_data = pd.read_parquet("data/test_triplets.parquet")

    # Group by query to get all positive passages for each query
    query_groups = test_data.groupby("query")["positive_passage"].agg(list)
    total_queries = len(query_groups)
    correct_predictions = 0

    # Test each unique query
    for query, positive_passages in query_groups.items():
        # Get model predictions
        predicted_docs, _ = inference.find_nearest_neighbors(query=query, k=k)

        # Check if any of the predictions match any positive passage
        for pred in predicted_docs:
            if pred in positive_passages:
                correct_predictions += 1
                break

    accuracy = (correct_predictions / total_queries) * 100
    print(f"Total unique queries tested: {total_queries}")
    print(f"Queries with at least one correct match: {correct_predictions}")
    print(f"Accuracy: {accuracy:.2f}%")
    return accuracy


def sanity_test(inference: Inference):
    print("Finding nearest neighbors...")
    for query in [
        "how to make coffee",
        "what is the capital of France",
        "best programming languages",
        "covid symptoms",
        "chocolate cake recipe",
    ]:
        docs, vals = inference.find_nearest_neighbors(query=query, k=3)
        print(f"\nQuery: {query}")
        print("Top 3 matches:")
        for doc, val in zip(docs, vals):
            print(f"Similarity: {val:.4f} | Doc: {doc}")
