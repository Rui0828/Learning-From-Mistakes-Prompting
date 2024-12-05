import faiss
import numpy
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

class EmbeddingModel:
    def __init__(self, model_name):
        # Setup the BERT model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)


    # 將句子轉換為嵌入向量
    def get_single_embedding(self, sentence):  # input type: str
        with torch.no_grad(): # Disable gradient calculation
            inputs = self.tokenizer(sentence, return_tensors='pt', truncation=True, padding=True)
            output = self.model(**inputs)
            embedding = output.last_hidden_state.mean(dim=1).cpu().numpy()[0]
        return embedding  # return type: dict

    # 批量轉換
    def get_multiple_embeddings(self, sentences):
        embeddings = {}
        for sentence in tqdm(sentences, desc="Loading datastore embeddings"):
            embeddings[sentence] = self.get_single_embedding(sentence)
        return embeddings

    # 將嵌入向量的list轉換為Faiss索引
    @staticmethod
    def embeddings2faiss(embeddings): # input type: list
        # 將嵌入向量轉換為Numpy數組
        np_embeddings = numpy.vstack(embeddings)
        #獲取嵌入維度並創建faiss索引
        faiss_index = faiss.IndexFlatL2(len(np_embeddings[0]))
        # 将数据添加到Faiss索引
        faiss_index.add(np_embeddings)
        return faiss_index

