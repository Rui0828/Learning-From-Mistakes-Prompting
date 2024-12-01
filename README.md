# Learning-From-Mistakes Prompting for Indigenous Language Translation
This repository contains the codebase for the method described in our paper: **"Learning-From-Mistakes Prompting for Indigenous Language Translation,"** presented at **LoResMT 2024**.

Our method, *Learning-From-Mistakes Prompting (LFM)*, introduces a feedback-driven framework to improve low-resource machine translation. By iteratively refining translations through error analysis and targeted adjustments, this approach significantly enhances the performance on indigenous language datasets.

For a detailed explanation of the methodology and experimental results, please refer to the full paper:  
- LoResMT@ACL 2024: [https://aclanthology.org/2024.loresmt-1.15/](https://aclanthology.org/2024.loresmt-1.15/)  
- arXiv: [https://arxiv.org/abs/2407.13343](https://arxiv.org/abs/2407.13343)

---

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Results](#results)
5. [Citation](#citation)
6. [License](#license)

---

## Introduction

Indigenous languages often face significant challenges in machine translation due to limited resources, including sparse datasets, complex linguistic structures, and diverse dialects. This repository provides the source code for reproducing the results in our paper, which proposes the following methods to address these challenges:

1. **KNN-Prompting with Retrieved Prompting Context (RPC):** Utilizes similar examples from a dataset to provide better context for translations.  
2. **Chain-of-Thought (CoT) Prompting:** Guides language models with step-by-step reasoning to improve translation accuracy.  
3. **Learning-From-Mistakes (LFM) Prompting:** Introduces feedback loops that iteratively refine translations based on identified errors.

Together, these methods work to enhance the translation of low-resource indigenous languages, making machine translation more accessible and accurate for these languages.

---

## Installation

### Option 1: Using Docker (Recommended)
1. Build the Docker image:  
    ```bash
    docker build -t lfm-prompting .  
    ```

2. Run the application:  
    ```bash
    docker run --rm -it lfm-prompting
    ```

3. (Optional) Use Docker Compose for more complex setups:
    ```bash
    docker-compose up
    ```

### Option 2: Manual Setup
1. Clone this repository:
    ```bash
    git clone https://github.com/Rui0828/Learning-From-Mistakes-Prompting.git
    cd Learning-From-Mistakes-Prompting
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. (Optional) Set up a virtual environment:
    ```bash
    python -m venv env
    source env/bin/activate  # On Windows: .\env\Scripts\activate
    ```

## Usage

### Docker Command
To process a sample input using Docker:
```bash
docker run --rm -it lfm-prompting python src/main.py --input examples/sample_input.txt --output results/output.txt
```

### Manual Command
Run the main script:
```bash
python src/main.py --input examples/sample_input.txt --output results/output.txt
```

## Results

The following table presents the translation results for **Southern Amis** across different methods. We report the performance metrics **BLEU1**, **BLEU2**, **BLEU3**, and **chrF++** for each method. The results demonstrate the effectiveness of different prompting techniques, with **LFM Prompting** achieving the best performance.

| **Methods**                                    | **BLEU1**$_{STD}$ | **BLEU2**$_{STD}$ | **BLEU3**$_{STD}$ | **chrF++**$_{STD}$ |
|------------------------------------------------|-------------------|-------------------|-------------------|--------------------|
| **Zeroshot**                                   | 1.0               | 0.0               | 0.0               | 3.9                |
| **20-shots**                                   | 18.0              | 4.9               | 1.9               | 16.3               |
| **Knn-Prompting (k=5)**                        | 30.1              | 14.4              | 6.9               | 28.1               |
| **Knn-Prompting (k=10)**                       | 33.3              | 16.4              | 8.0               | 34.2               |
| **Knn-Prompting w. RPC (k=5)**                 | 38.2$_{2.2}$      | 10.5$_{1.8}$      | 4.3$_{1.1}$       | 41.2$_{1.1}$       |
| **Knn-Prompting w. RPC (k=10)**                | 37.8$_{2.2}$      | 12.5$_{3.2}$      | 5.2$_{1.9}$       | 41.5$_{1.6}$       |
| **CoT Prompting**                              | 44.4$_{1.5}$      | 14.3$_{0.6}$      | 5.9$_{1.1}$       | 43.5$_{0.3}$       |
| **LFM Prompting**                              | 44.4$_{2.7}$      | 17.5$_{1.8}$      | 8.2$_{1.7}$       | 44.9$_{1.9}$       |

These results indicate that **LFM Prompting** provides a substantial improvement over traditional methods such as **Zeroshot** and **20-shots**, and also outperforms **Knn-Prompting** and **CoT Prompting** in terms of BLEU and chrF++ scores.

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

This work was conducted during my master's studies at the [**Natural Language Processing Lab, National Chung Hsing University (NCHU)**](https://nlpnchu.org). I would like to thank my advisor and colleagues for their guidance and support.


## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

The Apache License 2.0 allows you to freely use, modify, and distribute this software, with the following conditions:
- You must include a copy of the license in any distribution.
- You cannot use the name of the project or contributors without permission.
- Any modifications to the code must be documented.

For more details, please refer to the full license text available in the [LICENSE](LICENSE) file.
