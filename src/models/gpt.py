import os
import openai
from dotenv import load_dotenv
import configparser

# Load the environment variables
load_dotenv(".env")
api_key = os.getenv("OPENAI_API_KEY")

# Check if API key is loaded
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in the environment variables.")

openai.api_key = api_key

# Load the configuration file
config = configparser.ConfigParser()
if not config.read("translation/config.ini", encoding="utf-8"):
    raise FileNotFoundError("config.ini file not found.")

# Read the configuration settings with proper fallback values
model = config.get("gpt", "model", fallback="gpt-4")
max_output_tokens = config.getint("gpt", "max_output_tokens", fallback=512)
temperature = config.getfloat("gpt", "temperature", fallback=0.0)

print(f"Using GPT model: {model}")

# Define the GPT function
def gpt(messages):
    try:
        response = openai.chat.completions.create(
            model=model,
            messages=messages,
            max_completion_tokens=max_output_tokens,
            temperature=temperature,
            logprobs=True
        )
        # 取得生成的文本
        generated_text = response.choices[0].message.content.replace("Assistant:", "").strip()
        
        
        # 取得每個token的logit score
        token_logprobs = [i.logprob for i in response.choices[0].logprobs.content]
        # print(token_logprobs)

        # 計算平均logit score，作為整體的信心分數
        confidence_score = sum(token_logprobs) / len(token_logprobs)
        
        # print(f"Generated Text: {generated_text}")
        # print(f"Confidence Score (average logit): {confidence_score}")
        
        return generated_text, confidence_score
    
    except Exception as e:
        print(f"Error with OpenAI API request: {e}")
        return None, None




# Test
if __name__ == "__main__":
    # Test the GPT function
    
    # test_mes = [
    #     "What is the capital of the United States?", 
    #     "You are a Chinese-to-Amis language translator. Could you help me translate the following [zh] sentence: 中興大學是一所位於台灣的優質學校。"
    # ]
    
    # for mes in test_mes:
        
    #     print("-" * 100)
    #     print(f"Input message: {mes}")
        
    #     messages = [
    #         {"role": "user", "content": mes}
    #     ]
        
    #     response = gpt(messages)
        
    #     if response:
    #         print(f"Response: {response}")
    #     else:
    #         print("[ERROR]: Failed to get a response from the GPT model.")
        
    
    message = [{'role': 'user', 'content': "You are an Amis-to-Chinese language translator. The following are some [amis] to [zh] examples retrieved using Soundex code for phonetic matching:\n[amis]: Mafana' a romadiw kiso?\n[zh]: 你會唱歌嗎？\n\n[amis]: mafana' a milali'op kiso?\n[zh]: 你會洗臉嗎？\n\n[amis]: Talacowa kiso anodafak?\n[zh]: 你明天要去哪裡？\n\n[amis]: Talacowa kiso anini?\n[zh]: 你現在要去哪裡？\n\n[amis]: Talacowa anodafak kiso?\n[zh]: 你明天要去哪裡？\n\n[amis]: fainayan a kaka\n[zh]: 哥哥\n\n[*amis]: talacuwa\n[amis]: talacowa\n[zh]: 去哪裡/雖然\n\n[*amis]: talacuwa\n[amis]: talalosa\n[zh]: 含淚\n\n[*amis]: talacuwa\n[amis]: tolaku\n[zh]: 卡車\n\n[*amis]: talacuwa\n[amis]: tales\n[zh]: 鈍/鈍的\n\n[*amis]: talacuwa\n[amis]: teloc\n[zh]: 頂\n\n[*amis]: kisu\n[amis]: kako\n[zh]: 我\n\n[*amis]: kisu\n[amis]: kiso\n[zh]: 你\n\n[*amis]: kisu\n[amis]: koko\n[zh]: 雞\n\n[*amis]: kisu\n[amis]: kaka\n[zh]: 哥哥/姊姊\n\n[*amis]: kisu\n[amis]: kaeso\n[zh]: 很好吃\n\n[*amis]: kisu\n[amis]: kaesoay\n[zh]: 風味美食\n\n[*amis]: kisu\n[amis]: kayakay\n[zh]: 橋樑\n\n[*amis]: kisu\n[amis]: kiwkay\n[zh]: 教會\n\n[*amis]: kisu\n[amis]: kawas\n[zh]: 神/鬼/鬼魂\n\n[*amis]: kisu\n[amis]: kakaya\n[zh]: 長\n\n[*amis]: kisu\n[amis]: koskos\n[zh]: 抓癢\n\n[*amis]: kisu\n[amis]: kiwakiw\n[zh]: 攪動\n\n[*amis]: kisu\n[amis]: kohakohaw\n[zh]: 全是湯\n\n[*amis]: kisu\n[amis]: kasa\n[zh]: 各個\n\n[*amis]: kisu\n[amis]: kiyokay\n[zh]: 教會\n\n[*amis]: kisu\n[amis]: kaciw\n[zh]: 鵝\n\n[*amis]: kisu\n[amis]: kakohkoh\n[zh]: 眼鏡蛇\n\n[*amis]: kisu\n[amis]: kikoya\n[zh]: 氣球\n\n[*amis]: kisu\n[amis]: kakayaay\n[zh]: 長長的\n\n[*amis]: kisu\n[amis]: kaeso \n[zh]: 甘甜\n\n[*amis]: kisu\n[amis]: kakaya \n[zh]: 長\n\n[*amis]: kisu\n[amis]: kecec\n[zh]: 閉(眼睛)\n\n[*amis]: kisu\n[amis]: kosiyaw\n[zh]: 國民小學(漢語借詞)\n\n[*amis]: kisu\n[amis]: koki\n[zh]: 害怕\n\n[*amis]: anusani\n[amis]: anengan\n[zh]: 椅子\n\n\nThese examples contain both exact matches and phonetically similar matches. If you see [*amis], it means that the exact match couldn't be found, so I have retrieved the closest-sounding word based on fuzzy matching using the Soundex code.\n\nPlease help me determine whether the word matches the meaning of the [zh] translation. If it does, use the closest [amis] term. If it doesn't match, ignore the unmatched word and focus on the other terms. If you see English words, they represent proper nouns and should remain unchanged in the final translation.\n\nTranslate the following [amis] sentence using the retrieved examples. Use the provided examples as a guide, and where possible, choose the most relevant or correct words based on the context.\n\nInput sentence: A talacuwa kisu anusani\n\nPlease provide the translation only, without any additional explanation or commentary. Before providing the final translation, evaluate the coherence and appropriateness of the resulting sentence in the given context. Remove any redundant or unnecessary words to ensure a natural and fluent translation. Then, provide the translation only."}]
    
    response = gpt(message)
        
    if response:
        print(f"Response: {response}")
    else:
        print("[ERROR]: Failed to get a response from the GPT model.")
