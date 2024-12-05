import os
import json
import argparse
import random
import configparser
import warnings
from dotenv import load_dotenv
from src.models.knn import KNNRetriever
from src.models.callLLM import GPT
from src.utils.dataset_utils import DatasetUtils

# 忽略所有 FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

class Ch2AmisTranslator:
    def __init__(self, knn_k, findlexicon, sentences_path, lexicon_path, lexicon_embedding_path, sentences_emb, emb_model, rpc_prompt_path, cot_prompt_path, lfm_prompt_path, lfm_num, lfm_ict_num, llm_model):
        self.knn = KNNRetriever(sentences_path, lexicon_path, lexicon_embedding_path, sentences_emb, emb_model)
        self.knn_k = knn_k
        self.lfm_num = lfm_num
        self.lfm_ict_num = lfm_ict_num
        self.findlexicon = findlexicon
        self.sentences_emb = sentences_emb
        
        self.llm = llm_model
        
        self.rpc_prompt_path = rpc_prompt_path
        self.cot_prompt_path = cot_prompt_path
        self.lfm_prompt_path = lfm_prompt_path
        self.all_ch2amis = {value: key for key, value in json.load(open(sentences_path, "r", encoding="utf-8")).items()}
        
        self.LFM_in_context_examples = None

    def _get_rpc_prompt(self, input_sentence: str, examples: str) -> str:
        try:
            with open(self.rpc_prompt_path, 'r', encoding='utf-8') as file:
                prompt_content = file.read()
            return prompt_content.format(sentence=input_sentence, examples=examples)
        except FileNotFoundError:
            raise FileNotFoundError(f"Prompt 文件 {self.rpc_prompt_path} 找不到，請檢查文件是否存在。")
    
    def _get_cot_prompt(self, input_sentence: str, examples: str) -> str:
        try:
            with open(self.cot_prompt_path, 'r', encoding='utf-8') as file:
                cot_prompt = json.load(file)
            with open(self.rpc_prompt_path, 'r', encoding='utf-8') as file:
                prompt_content = file.read()
            cot_prompt.extend([{"role": "user", "content": prompt_content.format(sentence=input_sentence, examples=examples)}])
            return cot_prompt
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Prompt 文件 error，請檢查文件路徑是否正確。")
    
    
    # create wrong set for LFM
    def create_wrong_set(self, topN_sentences):
        wr = []
        for i in topN_sentences:
            #翻譯 (knn func 已排除找到原句的狀況)
            response = self.translate(i[0], "COT")
            wr.append([i[0], response, i[1]])
        return wr
    
    # find wrong example for LFM
    def find_wrong_example(self, sentence, k=2):
        # 找到與目標sentence最相似的k句話(預設k=2)
        topN_sentence = self.knn.find_knn_examples_topN_sentence(sentence, k)
        # print(topN_sentence)
        
        return self.create_wrong_set(topN_sentence)
    
    # get in-context-learning examples for LFM
    def get_incexp(self, k):
        pf = set()
        while len(pf) < k:
            temp = random.choice(list(self.sentences_emb.keys()))
            temp = (temp, self.all_ch2amis[temp])
            if temp not in pf:
                pf.add(temp)
            
        return self.create_wrong_set(pf)
    
    def _get_lfm_prompt(self, input_sentence: str, examples: str, init_translation: str, wrong_example: list) -> str:
        try:
            with open(self.lfm_prompt_path, 'r', encoding='utf-8') as file:
                prompt_content = json.load(file)
            
            # LFM hints
            hints = ""
            for i in wrong_example:
                if not hints:
                    hints += "[zh]:" + i[0] + " [your answer]:" + i[1] + " [correct answer]:" + i[2]
                else:
                    hints += " ; [zh]:" + i[0] + " [your answer]:" + i[1] + " [correct answer]:" + i[2]
            
            # LFM messages 
            messages = []
            # LFM in-context-learning examples
            for incexp in self.LFM_in_context_examples:
                messages.append({"role": "user", "content": prompt_content["COTLFM"].format(chinese=incexp[0], llm_answer=incexp[1])})
                messages.append({"role": "assistant", "content": incexp[2]})
            
            messages.append({"role": "system", "content": prompt_content["system"].format(hints=hints)})
            messages.append({"role": "user", "content": prompt_content["LFM"].format(examples=examples, chinese=input_sentence, llm_answer=init_translation)})
            
            return messages
        
        except FileNotFoundError:
            raise FileNotFoundError(f"Prompt 文件 {self.lfm_prompt_path} 找不到，請檢查文件是否存在。")
    
    def translate(self, sentence, mode):
        if mode == "RPC":
            examples = self.knn.find_knn_examples(sentence, self.knn_k, self.findlexicon)
            prompt_content = self._get_rpc_prompt(sentence, examples)
            response = self.llm.get_response(prompt_content)
            return response
        
        if mode == "COT":
            examples = self.knn.find_knn_examples(sentence, self.knn_k, self.findlexicon)
            prompt_content = self._get_cot_prompt(sentence, examples)
            response = self.llm.get_response(prompt_content)
            return response
        elif mode == "LFM":
            # get in-context-learning examples
            if self.LFM_in_context_examples is None:
                self.LFM_in_context_examples = self.get_incexp(self.lfm_ict_num)
            
            # get wrong example
            wrong_example = self.find_wrong_example(sentence, self.lfm_num)
            
            # get initial translation
            examples = self.knn.find_knn_examples(sentence, self.knn_k, self.findlexicon)
            init_translation = self.translate(sentence, "COT")
            
            prompt_content = self._get_lfm_prompt(sentence, examples, init_translation, wrong_example)
            response = self.llm.get_response(prompt_content)
            return response
            

def load_config(config_path="config.ini"):
    """Load configuration from the ini file."""
    config = configparser.ConfigParser()
    config.read(config_path, encoding="utf-8")
    return {
        "mode": config["ch2amis"]["mode"],
        "Knn_k": config.getint("ch2amis", "Knn_k", fallback=5),
        "LFM_num": config.getint("ch2amis", "LFM_num", fallback=2),
        "LFM_ICT_num": config.getint("ch2amis", "LFM_ICT_num", fallback=2),
        "findlexicon": config.get("ch2amis", "Find_lexicon", fallback=True),
        "emb_model":config.get("ch2amis", "embedding_model", fallback="DMetaSoul/sbert-chinese-general-v2"),
        "sentences_path": config['datapath']['sentences'],
        "lexicon_path": config['datapath']['lexicon'],
        "sentence_embedding_path": config['datapath']['sentence_embedding'],
        "lexicon_embedding_path": config['datapath']['lexicon_embedding'],
        "rpc_prompt_path": config['datapath']['rpc_prompt'],
        "cot_prompt_path": config['datapath']['cot_prompt'],
        "lfm_prompt_path": config['datapath']['lfm_prompt']
    }

def parse_arguments(defaults):
    """Parse command-line arguments, overriding config defaults."""
    parser = argparse.ArgumentParser(description="Chinese-to-Amis translations.")
    parser.add_argument("input_sentence", type=str, help="Input sentence to process.")
    parser.add_argument("--translation_mode", type=str, default=defaults["mode"], help="Translation mode (RPC, COT, LFM).")
    parser.add_argument("--Knn_k", type=int, default=defaults["Knn_k"], help="Number of nearest neighbors to retrieve.")
    parser.add_argument("--LFM_num", type=int, default=defaults["LFM_num"], help="Number of examples for LFM.")
    parser.add_argument("--LFM_ICT_num", type=int, default=defaults["LFM_ICT_num"], help="Number of in-context-learning examples for LFM.")
    parser.add_argument("--findlexicon", type=bool, default=defaults["findlexicon"], help="Whether to find lexicon.")
    parser.add_argument("--sentences_path", type=str, default=defaults["sentences_path"], help="Path to the sentences file.")
    parser.add_argument("--lexicon_path", type=str, default=defaults["lexicon_path"], help="Path to the lexicon file.")
    parser.add_argument("--sentence_embedding_path", type=str, default=defaults["sentence_embedding_path"], help="Path to the sentence embedding file.")
    parser.add_argument("--lexicon_embedding_path", type=str, default=defaults["lexicon_embedding_path"], help="Path to the lexicon embedding file.")
    parser.add_argument("--emb_model", type=str, default=defaults["emb_model"], help="Type of embedding model to use.")
    parser.add_argument("--rpc_prompt_path", type=str, default=defaults["rpc_prompt_path"], help="Path to the prompt file.")
    parser.add_argument("--cot_prompt_path", type=str, default=defaults["cot_prompt_path"], help="Path to the prompt file.")
    parser.add_argument("--lfm_prompt_path", type=str, default=defaults["lfm_prompt_path"], help="Path to the prompt file.")
    return parser.parse_args()


# Test
if __name__ == "__main__":

    # Load defaults from config
    config_defaults = load_config()
    
    # Parse arguments with overrides
    args = parse_arguments(config_defaults)
    
    # Extract parameters
    sentence = args.input_sentence
    translation_mode = args.translation_mode
    knn_k = args.Knn_k
    lfm_num = args.LFM_num 
    lfm_ict_num = args.LFM_ICT_num
    findlexicon = args.findlexicon
    sentences_path = args.sentences_path
    lexicon_path = args.lexicon_path
    sentence_embedding_path = args.sentence_embedding_path
    lexicon_embedding_path = args.lexicon_embedding_path
    emb_model = args.emb_model
    rpc_prompt_path = args.rpc_prompt_path
    cot_prompt_path = args.cot_prompt_path
    lfm_prompt_path = args.lfm_prompt_path
    
    # Initialize the translator
    data_utils = DatasetUtils(emb_model)
    all_ch2amis = data_utils._load_json(sentences_path, invert=True)
    sentence_embeddings = data_utils._load_embeddings(sentence_embedding_path, all_ch2amis)
    
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    llm_model = GPT(api_key, "gpt-4o")
    
    translator = Ch2AmisTranslator(knn_k, findlexicon, sentences_path, lexicon_path, lexicon_embedding_path, sentence_embeddings, emb_model, rpc_prompt_path, cot_prompt_path, lfm_prompt_path, lfm_num, lfm_ict_num, llm_model)
    
    
    
    print("Test the translation function")
    print("Input sentence:", sentence)
    print("Translation mode:", translation_mode)
    result  = translator.translate(sentence, translation_mode)
    print("Translate:", result)
    print("Test completed")
    
    
    # Test sentences examples:
    "中興大學是一所位於台灣的優質學校"
    "我們工作很勤勞。"