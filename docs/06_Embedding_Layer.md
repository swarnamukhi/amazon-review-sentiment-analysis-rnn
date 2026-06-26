# Chapter 6: Embedding Layer

## 6.1 Introduction

The output of the previous chapter was a matrix of padded integer sequences.

Example:

```
[12, 25, 7, 84, 0, 0, 0]
```

Although these values are numbers, they **do not represent the meaning of words**.

For example,

```
phone → 84

good → 15

excellent → 231
```

The numbers **84**, **15**, and **231** are simply identifiers assigned by the tokenizer. They do not indicate that "good" and "excellent" have similar meanings.

An Embedding Layer converts these integer IDs into dense numerical vectors that capture semantic relationships between words.

These dense vectors become the actual input to the RNN.

---

## 6.2 Why is the Embedding Layer Needed?

After tokenization we already have numbers.

Example

```
I love this phone

↓

[12,25,7,84]
```

Can we directly give these numbers to the RNN?

**No.**

The numbers represent only word IDs.

They do not contain any information about:

* Meaning
* Similarity
* Context
* Relationships between words

The RNN would incorrectly assume:

```
84 > 25
```

has mathematical meaning.

But actually,

```
phone = 84

love = 25
```

The numbers are simply labels.

Therefore, another representation is required.

The Embedding Layer learns meaningful numerical representations of words.

---

## 6.3 Understanding Token IDs

Suppose the tokenizer creates

```
good → 5

excellent → 8

bad → 25

phone → 91
```

Do these values mean

```
excellent > good ?
```

No.

They simply represent the order in which words were assigned IDs.

The tokenizer does not understand language.

It only creates a dictionary.

The Embedding Layer is responsible for learning the meaning of words.

---

## 6.4 What is an Embedding?

An embedding is a dense vector representation of a word.

Instead of representing a word using a single integer,

```
phone → 91
```

the Embedding Layer converts it into a vector.

Example

```
phone

↓

[0.14, -0.92, 0.33, 0.77, ..., 0.18]
```

If the embedding dimension is 128,

every word is represented by a vector containing **128 learned numerical values**.

These values are learned automatically during model training.

---

## 6.5 What Does the Embedding Layer Learn?

The Embedding Layer learns relationships between words.

Words that appear in similar contexts receive similar vector representations.

For example,

```
good

excellent

amazing

fantastic
```

will have embeddings that are close together in the embedding space.

Similarly,

```
bad

terrible

worst

awful
```

will also have similar embeddings.

This allows the model to understand that these words have related meanings, even though they have different token IDs.

---

## 6.6 Embedding Layer in This Project

The Embedding Layer used in this project is:

```python
Embedding(
    input_dim=MAX_FEATURES,
    output_dim=128,
    input_length=MAX_LEN
)
```

### Parameter Explanation

### input_dim

The size of the vocabulary.

If

```python
MAX_FEATURES = 5000
```

the Embedding Layer creates vectors for the top **5000 words**.

---

### output_dim

The number of features used to represent each word.

In this project,

```python
output_dim = 128
```

Every token is converted into a **128-dimensional vector**.

---

### input_length

The number of tokens in each padded review.

Since

```python
MAX_LEN = 100
```

every review contains exactly **100 tokens** before entering the Embedding Layer.

---

## 6.7 Input and Output Shape

Input from Padding Layer

```
(batch_size,100)
```

Example

```
(128,100)
```

Meaning:

* 128 reviews in one batch.
* 100 tokens in each review.

After the Embedding Layer,

the output becomes

```
(batch_size,100,128)
```

Example

```
(128,100,128)
```

Meaning:

* 128 reviews
* 100 tokens in each review
* Every token represented by 128 learned features

This three-dimensional tensor becomes the input to the RNN.

---

## 6.8 Chapter Summary

The Embedding Layer transforms token IDs into dense vector representations that capture semantic meaning. Unlike token IDs, these vectors allow the model to learn relationships between words. The output of the Embedding Layer is a three-dimensional tensor of shape `(batch_size, MAX_LEN, embedding_dimension)`, which serves as the input to the RNN.

