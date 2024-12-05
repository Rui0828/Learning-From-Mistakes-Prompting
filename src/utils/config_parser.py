import os
import argparse
import configparser
from dotenv import load_dotenv

def load_env():
    """Load environment variables from the .env file."""
    load_dotenv()
    return {"openai_api_key": os.getenv("OPENAI_API_KEY")}
    
def load_config(config_path="config.ini"):
    """Load configuration from the ini file."""
    config = configparser.ConfigParser()
    config.read(config_path, encoding="utf-8")
    return {
        "translation_mode": config["ch2amis"]["mode"],
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
        "lfm_prompt_path": config['datapath']['lfm_prompt'],
        "gpt_model": config['gpt']['model'],
        "gpt_temperature": config.getfloat('gpt', 'temperature', fallback=0.0),
        "gpt_max_output_tokens": config.getint('gpt', 'max_output_tokens', fallback=512),
        "batch_result_path": config['batch']['result_path']
    }

def get_combined_config(config_path="config.ini"):
    """Combine configurations from .env and config.ini."""
    config = load_config(config_path)
    env_vars = load_env()
    config["env"] = env_vars
    return config


def parse_arguments(defaults):
    """Parse command-line arguments, overriding config defaults."""
    parser = argparse.ArgumentParser(description="Chinese-to-Amis translations.")
    parser.add_argument("input_sentence", type=str, nargs="?", help="Input sentence to process.")
    parser.add_argument("--batch", action="store_true", help="Run in batch evaluation mode.")
    parser.add_argument("--translation_mode", type=str, default=defaults["translation_mode"], help="Translation mode (RPC, COT, LFM).")
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
    parser.add_argument("--batch_result_path", type=str, default=defaults["batch_result_path"], help="Path to the batch result file.")
    parser.add_argument("--openai_api_key", type=str, default=defaults["env"]["openai_api_key"], help="OpenAI API key.")
    parser.add_argument("--gpt_model", type=str, default=defaults["gpt_model"], help="OpenAI GPT model to use.")
    parser.add_argument("--gpt_temperature", type=float, default=defaults["gpt_temperature"], help="Temperature for GPT sampling.")
    parser.add_argument("--gpt_max_output_tokens", type=int, default=defaults["gpt_max_output_tokens"], help="Maximum number of tokens to output.")
    return parser.parse_args()


