import os
import re
import json
import numpy as np
from tqdm import tqdm
from src.models.embedding import EmbeddingModel


class DatasetUtils:
    def __init__(self, model_name):
        self.embedding_model = EmbeddingModel(model_name)
    
    @staticmethod
    def _load_json(path, invert=False):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {v: k for k, v in data.items()} if invert else data
    
    @staticmethod
    def _load_lexicon(path):
        with open(path, "r", encoding="utf-8") as f:
            lexicon = list(json.load(f).items())
        lexicon_list = [w[1] for w in lexicon]
        return lexicon, lexicon_list


    def _load_embeddings(self, path, keys):
        """
        Load embeddings from a file or generate them if the file does not exist.

        Args:
            path (str): The file path to save/load embeddings.
            keys (list): A list of keys to generate embeddings for if the file doesn't exist.

        Returns:
            dict: A dictionary mapping keys to their embeddings.
        """
        if os.path.exists(path):
            embeddings = np.load(path, allow_pickle=True).item()
            print(f"Loaded embeddings successfully from {path}. Number of entries: {len(embeddings)}.")
            return embeddings

        # Generate embeddings if the file does not exist
        print(f"File not found: {path}. Generating embeddings...")
        embeddings = self.embedding_model.get_multiple_embeddings(keys)  # Generate embeddings
        np.save(path, embeddings)  # Save embeddings to file
        print(f"Embeddings generated and saved to {path}. Number of entries: {len(embeddings)}.")
        return embeddings


    def _load_lexicon_embeddings(self, path, lexicon_list):
        """
        Load or generate embeddings for the lexicon and create a FAISS index.

        Returns:
            tuple: A tuple containing:
                - embeddings (list): The list of embeddings for the lexicon.
                - faiss_index: The FAISS index created from the embeddings.
                - mapping (list): A list of mappings from lexicon index to original positions.
        """
        if os.path.exists(path):
            embeddings = np.load(path, allow_pickle=True)
            print(f"Loaded lexicon embeddings successfully from {path}. Number of entries: {len(embeddings)}.")
        else:
            embeddings = [
                self.embedding_model.get_single_embedding(re.sub(r"\(.*?\)", "", lexicon).strip() or lexicon)
                for s_lexicon_list in tqdm(lexicon_list, desc="Loading lexicon embeddings")
                for lexicon in s_lexicon_list
            ]
            np.save(path, embeddings)
            print(f"Generated and saved lexicon embeddings to {path}. Number of entries: {len(embeddings)}.")

        # Create FAISS index and mapping
        faiss_index = self.embedding_model.embeddings2faiss(embeddings)
        mapping = [[i, j] for i, lst in enumerate(lexicon_list) for j in range(len(lst))]

        return faiss_index, mapping