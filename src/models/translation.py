import os
import json
import argparse
import configparser
import warnings
from dotenv import load_dotenv
from src.models.knn import KNNRetriever
from src.models.callLLM import GPT
from src.utils.dataset_utils import DatasetUtils

# 忽略所有 FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)



class Ch2AmisTranslator:
    def __init__(self, knn_k, findlexicon, sentences_path, lexicon_path, lexicon_embedding_path, emb_model, rpc_prompt_path, llm_model):
        self.knn = KNNRetriever(sentences_path, lexicon_path, lexicon_embedding_path, emb_model)
        self.knn_k = knn_k
        self.findlexicon = findlexicon
        
        self.llm = llm_model
        
        self.rpc_prompt_path = rpc_prompt_path
        self.all_ch2amis = {value: key for key, value in json.load(open(sentences_path, "r", encoding="utf-8")).items()}


    # def _get_prompt(self, input_sentence: str, examples: str) -> str:
    #     try:
    #         with open(self.prompt_path, 'r', encoding='utf-8') as file:
    #             prompt_content = file.read()
    #         return prompt_content.format(input_sentence=input_sentence, examples=examples)
    #     except FileNotFoundError:
    #         raise FileNotFoundError(f"Prompt 文件 {self.prompt_path} 找不到，請檢查文件是否存在。")


    # # Translation function
    # def translate(self, sentence, mode):
        
    #     if sentence in self.all_ch2amis:
    #         return self.all_ch2amis[sentence]
        
    #     # Get knn examples
    #     examples = self.knn.find_knn_examples(sentence, self.knn_k, self.findlexicon)
    #     cot_examples = self.knn.find_knn_examples_topN_sentence(sentence, self.cot_num)

    #     # Generate messages
    #     messages = []
    #     for i in cot_examples:
    #         # COT examples
    #         cot_examples_content = self.knn.find_knn_examples(i[0], self.knn_k, self.findlexicon)
    #         cot_prompt_content = self._get_prompt(i[0], cot_examples_content)
            
    #         messages.append({"role": "user", "content": cot_prompt_content})
    #         messages.append({"role": "assistant","content": assistant_response_format.format(i[1])})

    #     # Target sentence
    #     target_prompt_content = self._get_prompt(sentence, examples)
    #     messages.append({"role": "user", "content": target_prompt_content})
        
    #     # print(messages)
        
    #     # Generate translation
    #     response = self.llm.get_response(messages, json_mode=True)
    #     translation = response['translation']
        
    #     return translation

    def _get_rpc_prompt(self, input_sentence: str, examples: str) -> str:
        try:
            with open(self.rpc_prompt_path, 'r', encoding='utf-8') as file:
                prompt_content = file.read()
            return prompt_content.format(sentence=input_sentence, examples=examples)
        except FileNotFoundError:
            raise FileNotFoundError(f"Prompt 文件 {self.rpc_prompt_path} 找不到，請檢查文件是否存在。")
    
    def translate(self, sentence, mode, sentence_embedding):
        if mode == "RPC":
            examples = self.knn.find_knn_examples(sentence, self.knn_k, sentence_embedding, self.findlexicon)
            prompt_content = self._get_rpc_prompt(sentence, examples)
            response = self.llm.get_response(prompt_content)
            return response
        

def load_config(config_path="config.ini"):
    """Load configuration from the ini file."""
    config = configparser.ConfigParser()
    config.read(config_path, encoding="utf-8")
    return {
        "Knn_k": config.getint("ch2amis", "Knn_k", fallback=5),
        "findlexicon": config.get("ch2amis", "Find_lexicon", fallback=True),
        "emb_model":config.get("ch2amis", "embedding_model", fallback="DMetaSoul/sbert-chinese-general-v2"),
        "sentences_path": config['datapath']['sentences'],
        "lexicon_path": config['datapath']['lexicon'],
        "sentence_embedding_path": config['datapath']['sentence_embedding'],
        "lexicon_embedding_path": config['datapath']['lexicon_embedding'],
        "rpc_prompt_path": config['datapath']['rpc_prompt'],
    }

def parse_arguments(defaults):
    """Parse command-line arguments, overriding config defaults."""
    parser = argparse.ArgumentParser(description="Chinese-to-Amis translations.")
    parser.add_argument("input_sentence", type=str, help="Input sentence to process.")
    parser.add_argument("--Knn_k", type=int, default=defaults["Knn_k"], help="Number of nearest neighbors to retrieve.")
    parser.add_argument("--findlexicon", type=bool, default=defaults["findlexicon"], help="Whether to find lexicon.")
    parser.add_argument("--sentences_path", type=str, default=defaults["sentences_path"], help="Path to the sentences file.")
    parser.add_argument("--lexicon_path", type=str, default=defaults["lexicon_path"], help="Path to the lexicon file.")
    parser.add_argument("--sentence_embedding_path", type=str, default=defaults["sentence_embedding_path"], help="Path to the sentence embedding file.")
    parser.add_argument("--lexicon_embedding_path", type=str, default=defaults["lexicon_embedding_path"], help="Path to the lexicon embedding file.")
    parser.add_argument("--emb_model", type=str, default=defaults["emb_model"], help="Type of embedding model to use.")
    parser.add_argument("--rpc_prompt_path", type=str, default=defaults["rpc_prompt_path"], help="Path to the prompt file.")
    return parser.parse_args()


# Test
if __name__ == "__main__":

    # Load defaults from config
    config_defaults = load_config()
    
    # Parse arguments with overrides
    args = parse_arguments(config_defaults)
    
    # Extract parameters
    sentence = args.input_sentence
    knn_k = args.Knn_k
    findlexicon = args.findlexicon
    sentences_path = args.sentences_path
    lexicon_path = args.lexicon_path
    sentence_embedding_path = args.sentence_embedding_path
    lexicon_embedding_path = args.lexicon_embedding_path
    emb_model = args.emb_model
    prompt_path = args.rpc_prompt_path
    
    # Initialize the translator
    data_utils = DatasetUtils(emb_model)
    all_ch2amis = data_utils._load_json(sentences_path, invert=True)
    sentence_embeddings = data_utils._load_embeddings(sentence_embedding_path, all_ch2amis)
    
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    llm_model = GPT(api_key)
    
    translator = Ch2AmisTranslator(knn_k, findlexicon, sentences_path, lexicon_path, lexicon_embedding_path, emb_model, prompt_path, llm_model)
    
    
    
    print("Test the translation function")
    print("Input sentence:", sentence)
    result  = translator.translate(sentence, "RPC", sentence_embeddings)
    print("Translate:", result)
    print("Test completed")
    
    
    # Test sentences examples:
    "中興大學是一所位於台灣的優質學校"
    "我們工作很勤勞。"