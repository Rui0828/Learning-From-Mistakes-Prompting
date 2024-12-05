import json
import random
import src.utils.config_parser as config_parser
from src.models.callLLM import GPT
from src.models.knn import KNNRetriever
from src.utils.dataset_utils import DatasetUtils

class Ch2AmisTranslator:
    def __init__(self, args, sentences_emb):
        self.knn = KNNRetriever(args, sentences_emb)
        self.knn_k = args.Knn_k
        self.lfm_num = args.LFM_num
        self.lfm_ict_num = args.LFM_ICT_num
        self.findlexicon = args.findlexicon
        self.sentences_emb = sentences_emb
        
        self.llm = GPT(args.openai_api_key, args.gpt_model)
        
        self.rpc_prompt_path = args.rpc_prompt_path
        self.cot_prompt_path = args.cot_prompt_path
        self.lfm_prompt_path = args.lfm_prompt_path
        self.all_ch2amis = DatasetUtils(args.emb_model)._load_json(args.sentences_path, invert=True)
        
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
        else:
            raise ValueError(f"Translation mode {mode} not supported.")

# Test
if __name__ == "__main__":

    config_defaults = config_parser.get_combined_config()
    args = config_parser.parse_arguments(config_defaults)
    
    # Initialize the translator
    data_utils = DatasetUtils(args.emb_model)
    all_ch2amis = data_utils._load_json(args.sentences_path, invert=True)
    sentence_embeddings = data_utils._load_embeddings(args.sentence_embedding_path, all_ch2amis)
    
    translator = Ch2AmisTranslator(args, sentence_embeddings)
    
    if not args.input_sentence:
        raise ValueError("Please provide an input sentence.")
    
    print("Test the translation function")
    print("Input sentence:", args.input_sentence)
    print("Translation mode:", args.translation_mode)
    result  = translator.translate(args.input_sentence, args.translation_mode)
    print("Translate:", result)
    print("Test completed")
    
    
    # Test sentences examples:
    "中興大學是一所位於台灣的優質學校"
    "我們工作很勤勞。"