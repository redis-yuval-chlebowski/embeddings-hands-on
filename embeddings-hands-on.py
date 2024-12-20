import argparse
from contextlib import contextmanager
import redis
from redis import Redis
from sentence_transformers import SentenceTransformer, util
import time
import json
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch  # Import PyTorch
import numpy as np  # Import numpy

database_host = "localhost"
database_port = 6379
max_retries = 5
retry_delay = 2  # seconds

def connect_to_redis_with_retry(host, port, max_retries, retry_delay):
    for attempt in range(max_retries):
        try:
            client = Redis(host=host, port=port)
            client.ping()  # Test the connection
            return client
        except redis.exceptions.ConnectionError as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                raise e

@contextmanager
def cleanup_database():
    cleanup_client = connect_to_redis_with_retry(database_host, database_port, max_retries, retry_delay)
    cleanup_client.flushall()
    yield
    cleanup_client.flushall()

def construct_string():
    result = ""
    while len(result) < 16:
        letter = input("Enter a letter: ")
        count = int(input("Enter the number of times to add the letter: "))
        if len(result) + count > 16:
            choice = input("String will exceed 16 characters. Trim last letters or retry? (trim/retry): ")
            if choice == "trim":
                result += letter * (16 - len(result))
                break
            elif choice == "retry":
                continue
        else:
            result += letter * count
    return result

def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="String similarity using embeddings")
    parser.add_argument('--redis-host', type=str, default="localhost", help="Redis server hostname (default: localhost)")
    parser.add_argument('--redis-port', type=int, default=6379, help="Redis server port (default: 6379)")
    parser.add_argument('--embedding', type=str, help="Print the embedding of the specified letter repeated 16 times")
    parser.add_argument('--visualize', action='store_true', help="Visualize the embeddings using PCA")
    parser.add_argument('--plot-distance', action='store_true', help="Plot distances between the query string and the top 5 similar strings")
    parser.add_argument('--cosine', action='store_true', help="Use cosine similarity instead of Euclidean distance")
    args = parser.parse_args()

    with cleanup_database():
        # Predefined strings with 16 characters each
        strings = [
            "a" * 16,
            "b" * 16,
            "c" * 16,
            "d" * 16,
            "e" * 16,
            "f" * 16,
            "g" * 16,
            "h" * 16,
            "i" * 16,
            "j" * 16,
            "k" * 16,
            "l" * 16,
            "m" * 16,
            "n" * 16,
            "o" * 16,
            "p" * 16,
            "q" * 16,
            "r" * 16,
            "s" * 16,
            "t" * 16,
            "u" * 16,
            "v" * 16,
            "w" * 16,
            "x" * 16,
            "y" * 16,
            "z" * 16
        ]

        # Load the pre-trained sentence transformer model
        model = SentenceTransformer('all-MiniLM-L6-v2')

        # Generate embeddings for the predefined strings
        embeddings = model.encode(strings, convert_to_tensor=True)


        # Print the embedding of the specified letter if the argument is provided
        if args.embedding:
            letter_string = args.embedding * 16
            if letter_string in strings:
                index = strings.index(letter_string)
                print(f"String: {letter_string}, Embedding: {embeddings[index].tolist()}")
            else:
                print(f"Error: The letter '{args.embedding}' is not in the predefined strings.")
            exit()

        # Query string
        query_string = construct_string()
        query_embedding = model.encode(query_string, convert_to_tensor=True)

        # Compute similarity
        query_embedding_np = query_embedding.cpu().numpy()
        embeddings_np = embeddings.cpu().numpy()
        if args.cosine:
            similarities = [cosine_similarity(query_embedding_np, emb) for emb in embeddings_np]
            similarity_type = "cosine similarity"
            results = sorted(zip(strings, similarities), key=lambda x: x[1], reverse=True)  # Higher similarity is more similar
        else:
            distances = [euclidean_distance(query_embedding_np, emb) for emb in embeddings_np]
            similarity_type = "Euclidean distance"
            results = sorted(zip(strings, distances), key=lambda x: x[1])  # Lower distance is more similar

        # Sort and print the ranked results based on similarity
        print(f"\nQuery string: {query_string}")
        print(f"\nTop 5 similar strings based on {similarity_type}:")
        for s, score in results[:5]:
            if args.cosine:
                print(f"String: {s}, Similarity: {score:.4f}")
            else:
                print(f"String: {s}, Distance: {score:.4f}")

        # Visualize embeddings if the argument is provided
        if args.visualize:
            # Add the query string and its embedding to the list
            strings.append(query_string)
            embeddings = torch.cat((embeddings, query_embedding.unsqueeze(0)), dim=0)

            pca = PCA(n_components=2)
            embeddings_cpu = embeddings.cpu().numpy()  # Move embeddings to CPU
            reduced_embeddings = pca.fit_transform(embeddings_cpu)
            plt.figure(figsize=(10, 10))
            colors = plt.cm.get_cmap('tab20', len(strings))

            for i, label in enumerate(strings):
                x, y = reduced_embeddings[i]
                plt.scatter(x, y, color=colors(i))
                plt.text(x, y, label, fontsize=9)

            if args.plot_distance:
                query_index = len(strings) - 1
                for s, score in results[:5]:
                    target_index = strings.index(s)
                    color = colors(target_index)
                    plt.plot([reduced_embeddings[query_index, 0], reduced_embeddings[target_index, 0]],
                             [reduced_embeddings[query_index, 1], reduced_embeddings[target_index, 1]], color=color)
                    mid_x = (reduced_embeddings[query_index, 0] + reduced_embeddings[target_index, 0]) / 2
                    mid_y = (reduced_embeddings[query_index, 1] + reduced_embeddings[target_index, 1]) / 2
                    if args.cosine:
                        angle = np.arccos(cosine_similarity(query_embedding_np, embeddings_np[target_index])) * 180 / np.pi
                        plt.text(mid_x, mid_y - 0.1, f'Angle: {angle:.2f}°', color=color, fontsize=8)
                    else:
                        plt.text(mid_x, mid_y, f'{score:.4f}', color=color, fontsize=8)

            plt.title("PCA of String Embeddings")
            plt.xlabel("Principal Component 1")
            plt.ylabel("Principal Component 2")
            plt.show()
            exit()

        # Connect to Redis
        database_host = args.redis_host
        database_port = args.redis_port
        search_client = connect_to_redis_with_retry(database_host, database_port, max_retries, retry_delay)

        # Store the embeddings in Redis
        for i, s in enumerate(strings[:-1]):  # Exclude the query string
            search_client.hset(f"string:{s}", mapping={"string": s, "embedding": json.dumps(embeddings[i].cpu().tolist())})