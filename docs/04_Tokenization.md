# Chapter 4: Tokenization & Vocabulary Creation

## 4.1 Introduction

Natural Language Processing (NLP) models cannot directly understand human language. They require numerical input to perform mathematical operations during training.

Tokenization is the process of converting textual data into numerical representations by assigning a unique integer to every unique word in the dataset.

In this project, the **Keras Tokenizer** was used to build a vocabulary from the Amazon review dataset and convert every review into a sequence of integer IDs.

---

## 4.2 Why Tokenization is Required

Deep Learning models such as RNNs and LSTMs perform mathematical operations on numbers rather than text.

For example,

Original Review

```text
I love this phone
```

A neural network cannot process the above sentence directly.

After tokenization,

```text
I      → 12

love   → 25

this   → 7

phone  → 84
```

The sentence becomes

```text
[12, 25, 7, 84]
```

Now the review can be processed by the Embedding layer.

---

## 4.3 Tokenization Workflow

```text
Customer Reviews
        │
        ▼
Tokenizer
        │
        ▼
Build Vocabulary
        │
        ▼
Assign Integer IDs
        │
        ▼
Convert Reviews into Integer Sequences
```

---

## 4.4 Building the Vocabulary

The first step in tokenization is building the vocabulary.

The tokenizer scans every review in the training dataset and identifies all unique words.

Example

Dataset

```text
I love this phone

This phone is good

I love it
```

Vocabulary

```text
i

love

this

phone

is

good

it
```

Each unique word receives an integer index.

Example

```text
i      → 1

love   → 2

this   → 3

phone  → 4

is     → 5

good   → 6

it     → 7
```

The vocabulary is created **only from the training data**.

---

## 4.5 Keras Tokenizer Used in This Project

The following code was used to create the tokenizer.

```python
tokenizer = Tokenizer(num_words=MAX_FEATURES)
```

### Explanation

* **Tokenizer()** creates a tokenizer object.
* **num_words** limits the vocabulary size.
* Only the most frequent words are retained.
* Less frequent words are ignored during sequence generation.

This helps reduce memory usage and improves training efficiency.

---

## 4.6 Fitting the Tokenizer

```python
tokenizer.fit_on_texts(x_train)
```

### What happens internally?

When `fit_on_texts()` is executed, the tokenizer:

1. Reads every review.
2. Splits each review into words.
3. Counts how many times every word appears.
4. Builds the vocabulary.
5. Assigns an integer ID to every unique word.

No numerical sequences are created at this stage.

Only the vocabulary is learned.

---

## 4.7 Converting Text into Sequences

After the vocabulary is created,

```python
x_sequences = tokenizer.texts_to_sequences(x_train)
```

is executed.

Each word is replaced by its corresponding integer index.

Example

Original Review

```text
i love this phone
```

Vocabulary

```text
i → 1

love → 2

this → 3

phone → 4
```

Output

```text
[1, 2, 3, 4]
```

Now every review becomes a sequence of integers.

---

## 4.8 Important Tokenizer Attributes

### word_index

Stores the mapping

```text
Word → Integer
```

Example

```python
tokenizer.word_index
```

Output

```text
{
'i':1,
'love':2,
'phone':3
}
```

---

### index_word

Stores the reverse mapping

```text
Integer → Word
```

Example

```python
tokenizer.index_word
```

Output

```text
{
1:'i',
2:'love',
3:'phone'
}
```

---

### word_counts

Stores the frequency of every word.

Example

```python
tokenizer.word_counts
```

Output

```text
love : 1285

phone : 956

good : 1834
```

This information is used to rank words by frequency.

---

## 4.9 Output of Tokenization

Before Tokenization

```text
i love this phone
```

After Tokenization

```text
[12,25,7,84]
```

The output is **not yet ready** for the RNN because every review has a different length.

The next chapter explains how these sequences are converted into equal lengths using **Sequence Padding**.

---

## 4.10 Deep Dive

The tokenizer performs two independent tasks.

**Step 1**

Build the vocabulary.

```python
tokenizer.fit_on_texts()
```

**Step 2**

Convert text into sequences.

```python
tokenizer.texts_to_sequences()
```

These two operations should not be confused.

The tokenizer first learns the vocabulary and then uses that vocabulary to transform reviews into numerical sequences.

---

## 4.11 Interview Questions

**Q1. What is tokenization?**

**Q2. Why can't RNNs process raw text directly?**

**Q3. What is a vocabulary?**

**Q4. What is the difference between `fit_on_texts()` and `texts_to_sequences()`?**

**Q5. What is `word_index`?**

**Q6. What is `word_counts`?**

**Q7. Why is `num_words` used?**

---

## 4.12 Common Mistakes

* Calling `texts_to_sequences()` before `fit_on_texts()`.
* Creating a new tokenizer during prediction.
* Forgetting to save the tokenizer.
* Fitting the tokenizer on both training and test data, which causes data leakage.

---

## 4.13 Best Practices

* Fit the tokenizer only on the training data.
* Save the tokenizer after training.
* Use the same tokenizer during inference.
* Limit the vocabulary size using `num_words` when appropriate.

---

## 4.14 Chapter Summary

In this chapter, customer reviews were converted from raw text into sequences of integer IDs using the Keras Tokenizer. The tokenizer first built a vocabulary from the training data and then transformed each review into a numerical sequence. These sequences form the input to the next stage of the NLP pipeline, where they are converted to a fixed length using sequence padding.

