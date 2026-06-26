# Chapter 3: Data Preprocessing

## 3.1 Introduction

Data preprocessing is the process of cleaning and standardizing raw textual data before it is used to train a Deep Learning model.

Customer reviews collected from online platforms often contain inconsistent formatting, uppercase letters, HTML tags, unnecessary spaces, and other unwanted elements. These inconsistencies increase the vocabulary size and make it difficult for the model to learn meaningful patterns.

In this project, a simple preprocessing pipeline was implemented to standardize the review text before tokenization.

---

## 3.2 Why is Data Preprocessing Required?

Raw text cannot be directly used for training a Deep Learning model.

Preprocessing helps to:

* Standardize the text.
* Reduce unnecessary variations in words.
* Remove unwanted HTML formatting.
* Improve vocabulary consistency.
* Prepare clean text for tokenization.

By cleaning the text before tokenization, the tokenizer creates a more consistent vocabulary, allowing the model to learn more effectively.

---

## 3.3 Preprocessing Steps Implemented in This Project

The preprocessing pipeline used in this project consists of two steps.

```
Raw Amazon Review
        │
        ▼
Convert to Lowercase
        │
        ▼
Remove HTML <br /> Tags
        │
        ▼
Clean Review
```

These preprocessing steps were intentionally kept simple to demonstrate how an Embedding layer together with RNN/LSTM models can learn meaningful representations from relatively raw text.

---

## 3.4 Convert Text to Lowercase

Customer reviews may contain the same word written in different letter cases.

Example:

```
GOOD

Good

good
```

Without converting the text to lowercase, the tokenizer would treat these as three different words.

After preprocessing:

```
good

good

good
```

Converting all reviews to lowercase reduces the vocabulary size and ensures that identical words are represented consistently throughout the dataset.

---

## 3.5 Remove HTML Break Tags

The Amazon review dataset contains HTML line break tags such as:

```
<br />
```

These tags are used only for formatting and do not contribute to sentiment prediction.

Example:

Original Review

```
I love this phone.<br /><br />Excellent quality.
```

After preprocessing

```
i love this phone. excellent quality.
```

Removing HTML tags ensures that only meaningful textual information is passed to the tokenizer.

---

## 3.6 Preprocessing Function

The preprocessing function applies the cleaning steps to every customer review before tokenization.

The implementation performs the following operations:

1. Converts all text to lowercase using the `lower()` function.
2. Removes HTML `<br />` tags using the `replace()` function.

This preprocessing function is applied to every review in the dataset to ensure consistency before vocabulary creation.

---

## 3.7 Example

Original Review

```
I LOVE this Product!<br /><br />Amazing Quality
```

After Preprocessing

```
i love this product! amazing quality
```

Notice that punctuation is still present because punctuation removal was **not** implemented in this project.

---

## 3.8 Other Common Preprocessing Techniques

Many NLP projects perform additional preprocessing such as:

* Removing punctuation
* Removing URLs
* Removing numbers
* Removing special characters
* Stop word removal
* Stemming
* Lemmatization

These techniques were **not implemented in this project** because the objective was to build a simple preprocessing pipeline and allow the Embedding layer and recurrent neural networks to learn meaningful representations directly from the text.

---

## 3.9 Chapter Summary

In this chapter, the raw Amazon reviews were standardized by converting all text to lowercase and removing HTML `<br />` tags. These preprocessing steps reduced inconsistencies in the text and prepared the reviews for the next stage of the NLP pipeline: **Tokenization and Vocabulary Creation**.

