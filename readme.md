# Predicting Human Emotions with RoBERTa and XLNet

## Business Objective

Explore and leverage advanced language models to enhance sentiment prediction beyond BERT's capabilities. In this project, we focus on two architectures:

- RoBERTa: A Robustly Optimized BERT Pretraining Approach
- XLNet: Generalized Autoregressive Pretraining for Language Understanding

We use the architectures, investigate their training and optimization techniques, and apply them to classify human emotions into distinct categories.

---

## Data Description

The dataset, named "Emotion," comprises English Twitter messages annotated with six basic emotions: anger, fear, joy, love, sadness, and surprise. The dataset, sourced from the Hugging Face library, consists of three categories:

- Train: 16,000 rows, 2 columns
- Validation: 2,000 rows, 2 columns
- Test: 2,000 rows, 2 columns

The two columns represent labels and text, with labels corresponding to different emotions (0: sadness, 1: joy, 2: love, 3: anger, 4: fear, 5: surprise).

---

## Aim

The project aims to build and evaluate two emotion classification models: RoBERTa and XLNet.

---

## Tech Stack

- Language: `Python`
- Libraries: `datasets`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `ktrain`, `transformers`, `tensorflow`, `sklearn`

---

## Environment

- Jupyter Notebook
- Google Colab Pro (Recommended)

---

## Approach

1. **Install Required Libraries**
2. **Load 'Emotion' Dataset**
3. **Read Dataset Across Categories**
4. **Convert Dataset to Dataframe and Create a New Feature**
5. **Data Visualization**
   - Histogram Plots
6. **RoBERTa Model**
   - Create RoBERTa model instance
   - Split train and validation data
   - Perform Data Pre-processing
   - Compile RoBERTa in a K-train learner object
   - Find optimal learning rate
   - Fine-tune RoBERTa on the dataset
   - Evaluate performance metrics
   - Save RoBERTa model
   - Apply RoBERTa on test data and assess performance
7. **Understanding Autoregressive and Autoencoder Models**
8. **XLNet Model**
   - Load required libraries
   - Create XLNet model instance
   - Split train and validation data
   - Perform Data Pre-processing
   - Compile XLNet in a K-train learner object
   - Find optimal learning rate
   - Fine-tune XLNet on the dataset
   - Evaluate performance metrics
   - Save XLNet model
   - Apply XLNet on test data and assess performance

---

## Modular Code Overview

**Src Folder**
- **Engine.py**
- **ML_Pipeline Folder**

**ML_Pipeline Folder**
- Contains functions in different Python files, appropriately named, for each step. These functions are called inside the Engine.py file.

**Output Folder**
- Contains the best-fitted model trained for this data. This model can be loaded for future use without retraining. Note: The model is built on a subset of data; running Engine.py with the entire data retrains the models.

**Lib Folder**
- Contains the original IPython notebooks.


---
