Chapter 2 – Dataset & Exploratory Data Analysis (EDA).
2.1 Introduction
Before building any Machine Learning or Deep Learning model, it is important to understand the dataset. Exploratory Data Analysis (EDA) helps identify the structure, quality, and distribution of the data before model training.

In this project, EDA is performed to understand the Amazon Review dataset, identify missing values, duplicate records, and analyze the distribution of sentiment labels.

A well-understood dataset leads to better preprocessing decisions and improves the overall performance of the model.

2.2 About the Dataset
The project uses an Amazon Product Review dataset containing customer reviews and their corresponding sentiment labels.

Each record consists of:

• Review Text – Customer's written opinion about a product.

• Sentiment Label – Indicates whether the review is Positive or Negative.

The goal of the project is to learn the relationship between review text and its corresponding sentiment so that the model can accurately classify unseen customer reviews.

2.3 Why was this Dataset Selected?
This dataset was selected because it represents a real-world Natural Language Processing (NLP) problem.

Customer reviews contain unstructured text with different writing styles, making them ideal for training sequence models such as RNNs and LSTMs.

The dataset is also suitable for demonstrating the complete NLP pipeline, including text preprocessing, tokenization, sequence padding, embedding generation, and sequence classification.

summary:
In this chapter, the Amazon Review dataset was explored to understand its structure and quality.

EDA confirmed the dataset characteristics, identified potential data quality issues, and provided insights that guided subsequent preprocessing steps.

The next chapter focuses on Data Preprocessing, where the raw review text is cleaned and prepared for tokenization.
