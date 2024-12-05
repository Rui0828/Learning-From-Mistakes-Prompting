import os
import re
import json
import random
import warnings
from nltk.translate.bleu_score import sentence_bleu
import src.utils.config_parser as config_parser
from src.models.translation import Ch2AmisTranslator
from src.utils.dataset_utils import DatasetUtils
from src.models.embedding import EmbeddingModel
from tqdm import tqdm

# 忽略所有 FutureWarning
warnings.filterwarnings("ignore")

class BatchEvaluator:
    def __init__(self, args):
        
        self.language = args.language
        print("Translating to:", self.language)
        
        self.all_ch2amis = DatasetUtils(args.emb_model)._load_json(args.sentences_path, invert=True)
        self.test_num = args.batch_test_num 
        self.mode = args.translation_mode
        self.batch_result_path = args.batch_result_path
        self.version = self.create_result_foder(self.batch_result_path)
        self.args = args
        
        
    def create_result_foder(self, result_path):
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        # Check for existing folders and create the next one
        folder_index = 1
        while os.path.exists(os.path.join(result_path, f"{self.language}_v{folder_index}")):
            folder_index += 1
        next_folder = os.path.join(result_path, f"{self.language}_v{folder_index}")
        os.makedirs(next_folder)
        return next_folder

    def save_file(self, file_path, data):
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(data)
        print(f"File saved to {file_path}")
    
    def create_test_data(self, test_num = None):
        if test_num is None:
            test_num = self.test_num
        
        test_datastore = dict(self.all_ch2amis)
        
        test_data = {}
        with tqdm(range(test_num), desc="Creating test data") as pbar:
            for _ in pbar:
                while True:
                    temp = random.choice(list(test_datastore.keys()))
                    
                    # 如果去標點後的amis句子不足4個單詞，BLEU-4會是0，所以要避免這種情況
                    if len(re.findall(r"[\w]+", test_datastore[temp])) >= 4:
                        test_data[temp] = test_datastore[temp]
                        del test_datastore[temp]
                        break
        
        # Save the test data
        self.save_file(os.path.join(self.version, "test_data.json"), json.dumps(test_data, ensure_ascii=False, indent=4))
        self.save_file(os.path.join(self.version, "test_datastore.json"), json.dumps(test_datastore, ensure_ascii=False, indent=4))
        
        return test_data, test_datastore

    def evaluate(self, test_data, test_datastore_embeddings, mode = None):
        
        print("Evaluating test data version:", self.version)
        
        if mode is None:
            mode = self.mode
        
        print("Translation mode:", mode)
        
        # Initialize the translator
        translator = Ch2AmisTranslator(self.args, test_datastore_embeddings)
        
        # Evaluate the test data
        result = []
        avg_bleu = [0, 0, 0, 0]
        for key in tqdm(test_data, desc="Evaluating test data"):
            evaluation = {}
            evaluation["input_sentence"] = key
            evaluation["ground_truth"] = test_data[key]
            evaluation["translation"] = translator.translate(key, mode)
            evaluation["BLEU1"] = sentence_bleu([evaluation["ground_truth"].split()], evaluation["translation"].split(), weights=(1, 0, 0, 0))
            evaluation["BLEU2"] = sentence_bleu([evaluation["ground_truth"].split()], evaluation["translation"].split(), weights=(0.5, 0.5, 0, 0))
            evaluation["BLEU3"] = sentence_bleu([evaluation["ground_truth"].split()], evaluation["translation"].split(), weights=(0.33, 0.33, 0.33, 0))
            evaluation["BLEU4"] = sentence_bleu([evaluation["ground_truth"].split()], evaluation["translation"].split(), weights=(0.25, 0.25, 0.25, 0.25))
            result.append(evaluation)
            avg_bleu[0] += evaluation["BLEU1"]
            avg_bleu[1] += evaluation["BLEU2"]
            avg_bleu[2] += evaluation["BLEU3"]
            avg_bleu[3] += evaluation["BLEU4"]
        
        avg_bleu = [round(i / len(result), 4) for i in avg_bleu]
        
        result.append({"avg_BLEU1": avg_bleu[0], "avg_BLEU2": avg_bleu[1], "avg_BLEU3": avg_bleu[2], "avg_BLEU4": avg_bleu[3]})
        
        # Save the evaluation result
        self.save_file(os.path.join(self.version, f"{mode}_evaluation_result.json"), json.dumps(result, ensure_ascii=False, indent=4))
        
        return result

# Test
if __name__ == "__main__":
    
    config_defaults = config_parser.get_combined_config()
    args = config_parser.parse_arguments(config_defaults)
    
    # 如果沒有 --batch 並且 input_sentence 沒有提供，則報錯
    if not args.batch and args.input_sentence is None:
        raise ValueError("Please provide an input sentence when not not using --batch")
    
    if not args.batch:
        
        # Initialize the translator
        data_utils = DatasetUtils(args.emb_model)
        all_ch2amis = data_utils._load_json(args.sentences_path, invert=True)
        sentence_embeddings = data_utils._load_embeddings(args.sentence_embedding_path, all_ch2amis)
        
        translator = Ch2AmisTranslator(args, sentence_embeddings)
        
        if not args.input_sentence:
            raise ValueError("Please provide an input sentence.")
        
        print("Input sentence:", args.input_sentence)
        print("Translation mode:", args.translation_mode)
        print("GPT model:", args.gpt_model)
        result  = translator.translate(args.input_sentence, args.translation_mode)
        print("Translate:", result)
        
    else:
        evaluator = BatchEvaluator(args)
        
        # Create test data
        test_data, test_datastore = evaluator.create_test_data(args.batch_test_num)
        test_datastore_embeddings = EmbeddingModel(args.emb_model).get_multiple_embeddings(list(test_datastore.keys()))
        print(f"Embeddings generated successfully. Number of entries: {len(test_datastore_embeddings)}.")
        
        if args.translation_mode == "ALL":
            for mode in ["RPC", "COT", "LFM"]:
                print("-" * 100)
                result = evaluator.evaluate(test_data, test_datastore_embeddings, mode)
                print(f"Evaluation result for mode {mode}:")
                print("avg_BLEU1:", result[-1]["avg_BLEU1"])
                print("avg_BLEU2:", result[-1]["avg_BLEU2"])
                print("avg_BLEU3:", result[-1]["avg_BLEU3"])
                print("avg_BLEU4:", result[-1]["avg_BLEU4"])
        else:
            print("-" * 100)
            result = evaluator.evaluate(test_data, test_datastore_embeddings, args.translation_mode)
            print("Evaluation result:")
            print("avg_BLEU1:", result[-1]["avg_BLEU1"])
            print("avg_BLEU2:", result[-1]["avg_BLEU2"])
            print("avg_BLEU3:", result[-1]["avg_BLEU3"])
            print("avg_BLEU4:", result[-1]["avg_BLEU4"])
            print("Batch evaluation completed")
