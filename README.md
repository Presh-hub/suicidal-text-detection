# Suicidal and Non-Suicidal Text Classification

An academic Natural Language Processing (NLP) project that explores machine-learning and deep-learning approaches for classifying social-media text into **suicidal** and **non-suicidal** categories.

> **Important:** This repository is for academic and research purposes only. The models in this project are not clinical diagnostic tools and should not replace assessment or decisions by qualified mental-health professionals.

## Project Overview

The project investigates several approaches for binary text classification, beginning with classical machine learning and progressing to deep-learning and transformer-based methods.

The experiments include:

- TF-IDF feature extraction
- Logistic Regression
- Word2Vec embeddings
- Random Forest
- LSTM
- Transformer / Multi-Head Attention
- BERT fine-tuning

A separate baseline notebook also explores TF-IDF with Logistic Regression and Linear SVM, together with confusion-matrix and ROC/AUC evaluation.

## Objectives

The main objectives of the project are to:

1. Clean and preprocess social-media text.
2. Convert raw text into numerical representations suitable for machine learning.
3. Train and compare multiple NLP classification approaches.
4. Evaluate model performance using appropriate classification metrics.
5. Examine how increasingly contextual NLP methods perform on the binary classification task.

## Dataset

The project uses a labelled social-media text dataset with two classes:

- `suicide`
- `non-suicide`

The report describes a total of **232,074 text entries**, with **116,037 samples in each class**.

Because the dataset may have licensing, privacy, or distribution restrictions, it is recommended that the raw dataset **not be committed directly to this repository** unless its licence explicitly allows redistribution.

Expected local structure:

```text
data/
└── Suicide_Detection.csv


### Notebook Descriptions

**`01_baseline_logistic_svm.ipynb`**

Contains the baseline workflow:

- Data loading
- Text cleaning
- Train-test split
- TF-IDF feature extraction
- Logistic Regression
- Linear SVM
- Classification report
- Confusion matrix
- ROC curve and AUC analysis

**`02_advanced_nlp_models.ipynb`**

Contains the extended NLP workflow:

- Dataset analysis
- Text preprocessing
- Label encoding
- Train-test split
- TF-IDF + Logistic Regression
- Word2Vec + Random Forest
- LSTM
- Transformer / Attention
- BERT fine-tuning

## Methodology

### 1. Text Preprocessing

The project applies text-cleaning steps such as:

- Lowercase conversion
- URL removal
- Punctuation and special-character removal
- Whitespace normalization
- Stopword removal
- Tokenization

### 2. Classical Machine Learning

TF-IDF is used to convert text into sparse numerical features. Logistic Regression is used as a classical baseline classifier, and the baseline notebook also evaluates Linear SVM.

### 3. Word Embeddings

Word2Vec is used to learn dense word representations. Sentence-level vectors are produced from the learned embeddings and used with a Random Forest classifier.

### 4. LSTM

An LSTM neural network is used to model word order and sequential dependencies in text.

### 5. Transformer / Attention

A custom transformer-style text classifier uses embedding, Multi-Head Attention, pooling, and dense layers to capture contextual relationships.

### 6. BERT Fine-Tuning

A pretrained BERT-family sequence-classification model is fine-tuned using Hugging Face Transformers.

## Evaluation

The project uses several evaluation methods, including:

- Accuracy
- Precision
- Recall
- F1-score
- Classification report
- Confusion matrix
- ROC curve
- AUC score

According to the project report, the advanced experiments reported approximately:

| Model | Reported Accuracy |
|---|---:|
| Logistic Regression | 94% |
| Word2Vec-based model | 91% |
| LSTM | 95% |
| Transformer | 94% |
| BERT | 98% |

These values should be interpreted as results from the reported experimental setup and should be reproduced independently before being treated as general performance claims.

## Requirements

The notebooks use Python libraries including:

```text
pandas
numpy
matplotlib
seaborn
nltk
scikit-learn
gensim
tensorflow
datasets
transformers
torch
```

Install the dependencies with:

```bash
pip install -r requirements.txt
```

## Running the Project

### Option 1: Google Colab

The notebooks were developed with Google Colab-compatible code.

1. Open the notebook in Google Colab.
2. Mount Google Drive if the dataset is stored there.
3. Update the dataset path if necessary.
4. Install any missing packages.
5. Run the cells in order from top to bottom.

### Option 2: Local Jupyter Environment

Clone the repository:

```bash
git clone <your-repository-url>
cd suicidal-text-classification
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it.

**Windows:**

```bash
.venv\Scripts\activate
```

**macOS/Linux:**

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start Jupyter:

```bash
jupyter notebook
```

Then open a notebook from the `notebooks/` directory.

## Limitations

This project has several important limitations:

- Social-media text can contain slang, sarcasm, ambiguity, misspellings, and incomplete context.
- A binary label cannot represent the full complexity of a person's mental-health state.
- High test accuracy does not mean the model is suitable for clinical or real-world intervention.
- Dataset-specific patterns may not generalize to other platforms, populations, languages, or time periods.
- False positives and false negatives can both have serious consequences in a safety-sensitive domain.

For these reasons, any future real-world system would require expert oversight, careful validation, privacy safeguards, ethical review, and human decision-making.

## Future Work

Possible future extensions include:

- Multilingual text classification
- More rigorous external validation
- Error analysis by text type
- Model calibration and threshold analysis
- Explainability experiments
- Bias and fairness evaluation
- A carefully designed research demo interface
- Comparison with newer transformer architectures

