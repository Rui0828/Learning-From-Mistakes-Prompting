# Learning-From-Mistakes Prompting for Indigenous Language Translation
This repository contains the codebase for the method described in our paper: **"Learning-From-Mistakes Prompting for Indigenous Language Translation,"** presented at **LoResMT 2024**.  

Our method, *Learning-From-Mistakes Prompting (LFM)*, is a feedback-driven framework that enhances low-resource machine translation by iteratively refining translations based on error analysis and targeted adjustments. This significantly improves translation performance for indigenous language datasets.  

For more details, please refer to the paper:  
- LoResMT@ACL 2024: [View on ACL Anthology](https://aclanthology.org/2024.loresmt-1.15/)  
- arXiv: [Read on arXiv](https://arxiv.org/abs/2407.13343)

---

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Datasets](#Datasets)
4. [Usage](#usage)
5. [Results](#results)
6. [Citation](#citation)
7. [License](#license)

---

## Introduction

Machine translation for indigenous languages faces unique challenges such as limited datasets, complex linguistic structures, and diverse dialects. This repository provides the code to reproduce the results of our paper, proposing three key methods:  

1. **KNN-Prompting with Retrieved Prompting Context (RPC):** Enhances context by leveraging similar examples.  
2. **Chain-of-Thought (CoT) Prompting:** Improves translation accuracy through step-by-step reasoning.  
3. **Learning-From-Mistakes (LFM) Prompting:** Iteratively refines translations using feedback-driven error corrections.  

Together, these methods bridge gaps in low-resource translation, making machine translation more accessible and effective for indigenous languages.  

---

## Datasets  

Our framework is designed to be adaptable to any **low-resource language** where a **parallel corpus** can be obtained. A parallel corpus includes aligned linguistic data between a high-resource language (e.g., English or Chinese) and a low-resource language, which can range from sentence-level pairs to word-level mappings, such as bilingual dictionaries.  

While our experiments focus on datasets from indigenous languages sourced from [Klokah Platform](https://web.klokah.tw), including **Southern Amis**, **Coastal Amis**, and others, the method itself is **language-agnostic**. This ensures that it is broadly applicable to any low-resource language with minimal parallel resources.  

### Key Highlights:
- **Parallel Corpus Requirement:** A minimal set of sentence- or word-level alignments between a high-resource and a low-resource language.  
- **Broad Applicability:** Suitable for diverse low-resource languages with sparse datasets and linguistic challenges.  
- **Indigenous Language Focus:** While tested on indigenous languages, the framework is not limited to these; it is universally applicable to low-resource translation tasks.  

This flexibility enables the application of our framework beyond the tested datasets, contributing to the development of machine translation for underrepresented languages globally.  

---

## Installation

To get started, clone this repository:  
```sh
git clone https://github.com/Rui0828/Learning-From-Mistakes-Prompting.git
cd Learning-From-Mistakes-Prompting
```

### Requirements
- **Python:** >= 3.8  
- **Docker:** >= 20.10 (for Docker-based setups)  

### Installation Options
#### Option 1: Using Docker Compose (Recommended)
Build the Docker Compose environment:  
```sh
docker-compose build  
```

#### Option 2: Using Docker Directly
Build the Docker image:  
```sh
docker build -t lfm-prompting .  
```

#### Option 3: Manual Setup
1. Set up a virtual environment (optional but recommended):  
    ```sh
    python -m venv env  
    source env/bin/activate  # On Windows: .\env\Scripts\activate  
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt  
    ```

### Set up your OpenAI API key:  

 - Export the `OPENAI_API_KEY` environment variable before running the code:   

   ```sh
   export OPENAI_API_KEY="your_api_key"  
   ```

 - Alternatively, create a `.env` file in the project root with the following content:  

   ```
   OPENAI_API_KEY=your_api_key  
   ```

## Usage
You can run the program in two modes:  
1. **Single Sentence Translation:**
   Translate a single Chinese sentence to the target language.  
2. **Batch Translation and Evaluation:**
   Use the `--batch` option to automatically split data into a test set (default: 100 sentences) and a datastore, perform batch translations, and evaluate results using BLEU scores. Results are saved in the `./results` directory.  

### Commands
#### Using Docker Compose (Recommended)
- **Single Translation:**
    ```sh  
    docker-compose run lfm-prompting "{input chinese sentence}"  
    ```
- **Batch Translation and Evaluation:**
    ```sh  
    docker-compose run lfm-prompting --batch  
    ```

#### Using Docker
- **Single Translation**
    ```sh  
    docker run --rm -v "$(pwd):/app" -w /app lfm-prompting "{input chinese sentence}"  
    ```
- **Batch Translation and Evaluation:**
    ```sh  
    docker run --rm -v "$(pwd):/app" -w /app lfm-prompting --batch  
    ```

#### Using Manual Setup
- **Single Translation**
    ```sh  
    python -m src.main "{input chinese sentence}"  
    ```
- **Batch Translation and Evaluation:**
    ```sh  
    python -m src.main --batch  
    ```


## Results

Below are the results for **Southern Amis** translation using different methods. Metrics include **BLEU** and **chrF++**. Our **LFM Prompting** outperforms all other approaches.  

| **Methods**                                    | **BLEU1 (STD)** | **BLEU2 (STD)** | **BLEU3 (STD)** | **chrF++ (STD)** |
|------------------------------------------------|-----------------|-----------------|-----------------|------------------|
| **Zeroshot**                                   | 1.0             | 0.0             | 0.0             | 3.9              |
| **20-shots**                                   | 18.0            | 4.9             | 1.9             | 16.3             |
| **Knn-Prompting (k=5)**                        | 30.1            | 14.4            | 6.9             | 28.1             |
| **Knn-Prompting (k=10)**                       | 33.3            | 16.4            | 8.0             | 34.2             |
| **Knn-Prompting with RPC (k=5)**               | 38.2 ± 2.2      | 10.5 ± 1.8      | 4.3 ± 1.1       | 41.2 ± 1.1       |
| **Knn-Prompting with RPC (k=10)**              | 37.8 ± 2.2      | 12.5 ± 3.2      | 5.2 ± 1.9       | 41.5 ± 1.6       |
| **CoT Prompting**                              | 44.4 ± 1.5      | 14.3 ± 0.6      | 5.9 ± 1.1       | 43.5 ± 0.3       |
| **LFM Prompting**                              | 44.4 ± 2.7      | 17.5 ± 1.8      | 8.2 ± 1.7       | 44.9 ± 1.9       |

> Note: **LFM Prompting** achieves the best **BLEU** and **chrF++** scores, demonstrating its effectiveness in refining translations.  

## Troubleshooting
If you encounter issues, here are some common solutions:  
- **Missing API Key:** Ensure your OpenAI API key is correctly set as an environment variable or in a `.env` file.  
- **Dependency Errors:** Verify Python dependencies using `pip install -r requirements.txt`. Consider using a virtual environment.  
- **Docker Errors:** Ensure Docker is installed and running. Check your Docker version compatibility.  

## Citation
If you use this code in your research, please cite our paper:  
```bibtex  
@inproceedings{liao2024learning,
  title={Learning-From-Mistakes Prompting for Indigenous Language Translation},
  author={Liao, You Cheng and Yu, Chen-Jui and Lin, Chi-Yi and Yun, He-Feng and Wang, Yen-Hsiang and Li, Hsiao-Min and Fan, Yao-Chung},
  booktitle={Proceedings of the The Seventh Workshop on Technologies for Machine Translation of Low-Resource Languages (LoResMT 2024)},
  pages={146--158},
  year={2024}
}  
```

## Acknowledgments

This work was conducted during my undergraduate studies at the [**Natural Language Processing Lab, National Chung Hsing University (NCHU)**](https://nlpnchu.org). I would like to thank my advisor and colleagues for their invaluable guidance and support throughout this project.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.  
