# Chapter 5: Sequence Padding

## 5.1 Introduction

After tokenization, every customer review is converted into a sequence of integer IDs. However, not all reviews have the same number of words. Some reviews may contain only a few words, while others may contain hundreds.

Deep Learning models such as RNNs and LSTMs expect every input sequence within a batch to have the same length. Therefore, before training the model, all review sequences must be converted to a fixed length.

This process is called **Sequence Padding**.

---

## 5.2 Why is Sequence Padding Required?

Consider the following reviews after tokenization.

Review 1

```text
[12, 25, 7]
```

Review 2

```text
[5, 14, 9, 3, 18, 45]
```

Review 3

```text
[6, 8]
```

Each review has a different length.

Since Deep Learning models process data in batches, every sequence within a batch must have the same shape.

Sequence Padding ensures that all reviews contain the same number of tokens.

---

## 5.3 Padding Workflow

```text
Raw Integer Sequences
        │
        ▼
Choose MAX_LEN
        │
        ▼
Pad Short Reviews
        │
        ▼
Truncate Long Reviews
        │
        ▼
Equal Length Sequences
```

---

## 5.4 Keras Implementation

The following function was used in this project.

```python
x_padded = pad_sequences(
    x_sequences,
    maxlen=MAX_LEN,
    padding='post',
    truncating='post'
)
```

The `pad_sequences()` function is provided by **Keras** and converts variable-length sequences into fixed-length sequences.

---

## 5.5 Understanding Each Parameter

### maxlen

Defines the maximum number of tokens allowed in each review.

Example

```python
MAX_LEN = 100
```

Every review will contain exactly **100 tokens** after padding.

---

### padding='post'

Short reviews are padded by adding zeros **at the end**.

Example

Original

```text
[12,25,7]
```

After Padding

```text
[12,25,7,0,0,0,0,0]
```

---

### truncating='post'

Reviews longer than `MAX_LEN` are truncated by removing tokens **from the end**.

Example

Original

```text
[1,2,3,4,5,6,7,8]
```

Suppose

```python
MAX_LEN = 5
```

After Truncation

```text
[1,2,3,4,5]
```

---

## 5.6 Why was 'post' used?

In this project,

```python
padding='post'
truncating='post'
```

were selected.

Padding at the end preserves the original beginning of the review.

Similarly, truncating from the end keeps the initial part of the sentence intact.

This approach is commonly used in NLP tasks where the beginning of the sentence often contains important contextual information.

---

## 5.7 Shape Before and After Padding

Before Padding

```text
Review 1 → 15 tokens

Review 2 → 63 tokens

Review 3 → 92 tokens
```

After Padding

```text
Review 1 → 100 tokens

Review 2 → 100 tokens

Review 3 → 100 tokens
```

The dataset now has a consistent shape that can be processed efficiently by the Embedding layer.

---

## 5.8 Output Shape

Suppose:

* Number of Reviews = 25,000
* MAX_LEN = 100

After padding,

the dataset shape becomes

```text
(25000, 100)
```

Meaning:

* 25,000 reviews
* 100 tokens per review

This is the input to the Embedding layer.

---

## 5.9 Deep Dive

Padding does **not** change the meaning of the review.

The added zeros are simply placeholders used to make all sequences the same length.

The Embedding layer will later learn an embedding for valid word IDs, while padding tokens are treated separately during sequence processing.

---

## 5.10 Interview Questions

**Q1. Why is sequence padding required?**

**Q2. Why can't an RNN accept sequences of different lengths within the same batch?**

**Q3. What does `MAX_LEN` represent?**

**Q4. What is the difference between `padding='pre'` and `padding='post'`?**

**Q5. What is the difference between `truncating='pre'` and `truncating='post'`?**

**Q6. Which Keras utility is used for sequence padding?**

---

## 5.11 Common Mistakes

* Choosing a very small `MAX_LEN`, causing excessive truncation.
* Choosing an unnecessarily large `MAX_LEN`, increasing computation time.
* Using different padding strategies during training and prediction.
* Forgetting to pad new reviews before making predictions.

---

## 5.12 Best Practices

* Select `MAX_LEN` based on the review length distribution.
* Use the same padding strategy during both training and inference.
* Keep preprocessing, tokenization, and padding consistent throughout the project.

---

## 5.13 Chapter Summary

After tokenization, reviews had different lengths. Sequence Padding standardized every review to a fixed length of **MAX_LEN** tokens, allowing the data to be processed efficiently by the Embedding layer. The output of this step is a two-dimensional matrix of shape `(number_of_reviews, MAX_LEN)`, which becomes the input to the Embedding layer in the next chapter.

