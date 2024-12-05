import src.utils.config_parser as config_parser
from src.models.translation import Ch2AmisTranslator
from src.utils.dataset_utils import DatasetUtils

class BatchEvaluator:
    def __init__(self, args):
        pass
        

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
        
        