import re
import numpy as np
import jieba
import string
import configparser
from src.models.embedding import EmbeddingModel
from src.utils.dataset_utils import DatasetUtils


class KNNRetriever:
    def __init__(self, sentences_path, lexicon_path, sentence_embedding_path, lexicon_embedding_path, model_name):
        self.embedding_model = EmbeddingModel(model_name)
        self.sentences_path = sentences_path
        self.lexicon_path = lexicon_path
        self.sentence_embedding_path = sentence_embedding_path
        self.lexicon_embedding_path = lexicon_embedding_path
        
        
        self.data_utils = DatasetUtils(model_name)
        self.all_ch2amis = self.data_utils._load_json(self.sentences_path, invert=True)
        self.lexicon, self.lexicon_list = self.data_utils._load_lexicon(self.lexicon_path)
        self.sentence_embeddings = self.data_utils._load_embeddings(self.sentence_embedding_path, self.all_ch2amis.keys())
        self.faiss_index, self.lexicon_mapping = self.data_utils._load_lexicon_embeddings(self.lexicon_embedding_path, self.lexicon_list)
        
    # find_similar
    def find_similar(self, term):
        embedding = self.embedding_model.get_single_embedding(term)
        _, indices = self.faiss_index.search(np.array([embedding]), k=1)
        return self.lexicon_mapping[int(indices[0][0])]

    # _find_longest_match    
    def _find_longest_match(self, sentence):
        temp_list = []
        max_len = 0
        for e, w in enumerate(self.lexicon_list):
            for e2, w2 in enumerate(w):
                if len(sentence) >= len(w2) and w2 == sentence[:len(w2)]:
                    temp_list.append([e, e2])
                    max_len = max(max_len, len(w2))

        # 檢查是否有空字串資料
        if temp_list and max_len == 0:
            raise ValueError("Data Error: lexicon_list contains empty string elements.")
        
        return [item for item in temp_list if len(self.lexicon_list[item[0]][item[1]]) == max_len]


    # format_parallel_data
    def _format_parallel_data(self, data_idx):
        return f"[zh]: {'/'.join(self.lexicon_list[data_idx[0]])}\n[amis]: {self.lexicon[data_idx[0]][0]}\n\n"

    # check_not_found_sentence
    def _check_not_found_sentence(self, sentence):
        if not sentence:
            return ""
        
        seg_list = jieba.cut(sentence, cut_all=False)
        return ''.join(
            f"[*zh]: {text}\n[zh]: {'/'.join(self.lexicon_list[e[0]])}\n[amis]: {self.lexicon[e[0]][0]}\n\n"
            for text in seg_list if text.strip() and (e := self.find_similar(text))
        )

    # 查找詞表
    def trans(self, sentence):
        ans = ""
        cant_find_sentence = ""
        while sentence:
            index_list = self._find_longest_match(sentence)
            if index_list:
                ans += self._check_not_found_sentence(cant_find_sentence)
                cant_find_sentence = ""
                ans += ''.join(self._format_parallel_data(index) for index in index_list)
                sentence = sentence[len(self.lexicon_list[index_list[0][0]][index_list[0][1]]):]
            else:
                cant_find_sentence += sentence[0]
                sentence = sentence[1:]
        ans += self._check_not_found_sentence(cant_find_sentence)
        return ans

    # find_knn_examples_topN_sentence
    def find_knn_examples_topN_sentence(self, sentence, k):
        
        if k == 0:
            return []
        
        cp_datastore_embeddings = self.sentence_embeddings.copy()
        if sentence in cp_datastore_embeddings:
            del cp_datastore_embeddings[sentence]

        datastore_index = self.embedding_model.embeddings2faiss(list(cp_datastore_embeddings.values()))
        
        _, indices = datastore_index.search(np.array([self.embedding_model.get_single_embedding(sentence)]), k)

        return [[list(cp_datastore_embeddings.keys())[index], self.all_ch2amis[list(cp_datastore_embeddings.keys())[index]]]
                for index in indices[0]]

    # find_lexicon
    def _find_lexicon(self, sentence):
        translator = str.maketrans("，。！", "   ", string.punctuation)
        sentence_cleaned = sentence.translate(translator)

        # 去除英文字，查找中文詞表
        examples = self.trans(re.sub(r'[a-zA-Z]+', '', sentence_cleaned))

        # 英文詞直接保留
        for lexicon in re.findall(r'[a-zA-Z]+', sentence_cleaned):
            examples += f"[zh]: {lexicon}\n[amis]: {lexicon}\n\n"
        
        return examples

    
    # knn主程式
    def find_knn_examples(self, sentence, k, findlexicon=True):
        examples = ''.join(
            f"[zh]: {zh_example}\n[amis]: {amis_example}\n\n"
            for zh_example, amis_example in self.find_knn_examples_topN_sentence(sentence, k)
        )

        if findlexicon:
            examples += self._find_lexicon(sentence)

        return examples



# Test
if __name__ == "__main__":
    print("Test the knn function") 
    sentence = "中興大學是一所位於台灣的優質學校"
    
    # Load the configuration
    config = configparser.ConfigParser()
    config.read("config.ini", encoding="utf-8")
    sentences_path = config['translation.datapath']['sentences']
    lexicon_path = config['translation.datapath']['lexicon']
    sentence_embedding_path = config['translation.datapath']['sentence_embedding']
    lexicon_embedding_path = config['translation.datapath']['lexicon_embedding']
    model_name = config['translation.ch2amis']['embedding_model']
    
    knn = KNNRetriever(sentences_path, lexicon_path, sentence_embedding_path, lexicon_embedding_path, model_name)
    
    print(knn.find_knn_examples(sentence, 10))
    print("Test Done.")
