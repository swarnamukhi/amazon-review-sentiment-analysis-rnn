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

## 6.9 Internal Working of the Embedding Layer

The Embedding Layer does much more than simply replace token IDs with vectors.

When the model is created, Keras automatically builds a large matrix called the **Embedding Matrix**.

This matrix stores the vector representation of every word in the vocabulary.

During training, these vectors are continuously updated so that words with similar meanings obtain similar numerical representations.

The Embedding Layer is therefore a trainable lookup table rather than a fixed dictionary.

---

## 6.10 The Embedding Matrix

Suppose our project has the following configuration.

```python
MAX_FEATURES = 5000
output_dim = 128
```

When the model is built,

```python
Embedding(
    input_dim=5000,
    output_dim=128
)
```

Keras automatically creates an embedding matrix of size

```text
5000 × 128
```

This means

* 5000 rows
* 128 columns

Each row represents one word in the vocabulary.

Each column represents one learned feature of that word.

Graphically,

```
                128 Features
        ---------------------------------->
Word 1   [0.12  -0.44  0.88  ... 0.31]

Word 2   [0.56   0.22 -0.18 ... 0.90]

Word 3   [-0.75  0.40  0.67 ... -0.11]

.

.

.

Word 5000
```

Every vocabulary word owns one complete row.

---

## 6.11 Why is the Matrix 5000 × 128?

Many beginners think the matrix should be

```text
100 × 128
```

because every review contains 100 tokens.

This is incorrect.

The value

```text
5000
```

comes from the vocabulary size.

The value

```text
128
```

comes from the embedding dimension.

The embedding matrix must contain a vector for **every word in the vocabulary**, not every word in a sentence.

Therefore,

```
Embedding Matrix Shape

(Vocabulary Size × Embedding Dimension)

↓

5000 × 128
```

---

## 6.12 What Happens When a Review Enters the Embedding Layer?

Suppose a review after padding is

```text
[12, 25, 7, 84, 0, 0, 0]
```

These numbers are not used directly by the RNN.

Instead, each number is treated as the row index of the embedding matrix.

For example,

Token

```text
12
```

means

```
Read Row 12
```

Token

```text
25
```

means

```
Read Row 25
```

Token

```text
84
```

means

```
Read Row 84
```

The Embedding Layer simply retrieves those rows.

---

## 6.13 Example

Suppose

```
Embedding Matrix

Row 12

[0.12, -0.88, 0.33, .... 128 values]

Row 25

[-0.22, 0.64, -0.51, .... 128 values]

Row 7

[0.78, 0.19, -0.44, .... 128 values]
```

Input

```text
[12,25,7]
```

Output

```
[
 Vector for Token 12,

 Vector for Token 25,

 Vector for Token 7
]
```

The token IDs disappear.

Only vectors remain.

---

## 6.14 Why Does One Sentence Become 100 × 128?

Suppose

```
MAX_LEN = 100
```

Every review contains exactly

```
100 tokens
```

Each token is converted into

```
128 features
```

Therefore

```
100 Tokens

↓

100 Embedding Vectors

↓

100 × 128
```

This is why the Embedding Layer outputs a matrix of shape

```text
100 × 128
```

for every review.

This output is then passed to the RNN.

---

## 6.15 Batch Processing

Suppose

```python
batch_size = 128
```

The model processes

```
128 reviews simultaneously.
```

Before the Embedding Layer

```
(128,100)
```

Meaning

```
128 Reviews

100 Tokens each
```

After the Embedding Layer

```
(128,100,128)
```

Meaning

```
128 Reviews

↓

100 Tokens

↓

128 Features per Token
```

This three-dimensional tensor becomes the input to the RNN layer.

---

## 6.16 Important Observation

The Embedding Layer does **not** process one review at a time.

Instead, it processes an entire batch.

For a batch size of 128,

all first tokens are converted,

all second tokens are converted,

and so on,

using highly optimized matrix operations.

Although this happens internally in parallel, conceptually you can think of every token in every review being replaced by its corresponding embedding vector.

---

## 6.17 Summary

The Embedding Layer acts like a trainable lookup table.

It maintains an embedding matrix whose size is determined by the vocabulary size and embedding dimension.

Each token ID is used to retrieve one row from this matrix.

For one review containing 100 tokens, the output becomes a matrix of shape

```text
100 × 128
```

For a batch of 128 reviews, the output becomes

```text
128 × 100 × 128
```

This tensor is passed directly to the RNN layer.
## 6.18 How Does the Embedding Layer Learn?

One of the most common interview questions is:

**"Where do the embedding vectors come from?"**

The answer is:

**The Embedding Layer learns them automatically during model training.**

Unlike the tokenizer, which only assigns integer IDs, the Embedding Layer contains trainable parameters that are continuously updated while the model is being trained.

---

## 6.19 Random Initialization

When the model is created,

```python
Embedding(
    input_dim=5000,
    output_dim=128
)
```

Keras first creates the embedding matrix.

Initially, every row contains random values.

Example

```text
Word 1

[0.14, -0.32, 0.76, ....]

Word 2

[-0.81, 0.19, 0.42, ....]

Word 3

[0.67, -0.14, 0.55, ....]
```

These values have **no meaning**.

They are simply random numbers generated during model initialization.

At this stage, the model knows nothing about language.

---

## 6.20 Forward Pass

Suppose one review after padding is

```text
[12,25,7,84]
```

The Embedding Layer retrieves the corresponding rows from the embedding matrix.

```text
Token 12

↓

Embedding Vector

↓

[0.18, -0.24, 0.56, ....]

Token 25

↓

Embedding Vector

↓

[-0.67, 0.91, -0.13, ....]

Token 7

↓

Embedding Vector

↓

[0.42, -0.58, 0.77, ....]
```

These vectors become the input to the RNN.

The RNN processes these vectors and produces a prediction.

Example

```text
Predicted Sentiment

Positive

Probability = 0.62
```

---

## 6.21 Loss Calculation

The prediction produced by the model is compared with the actual sentiment label.

Example

```text
Actual Label

Positive

↓

1

Predicted Probability

0.62
```

The Binary Crossentropy loss function measures how far the prediction is from the correct answer.

If the prediction is inaccurate, the loss value becomes larger.

A larger loss indicates that the model needs to improve its parameters.

---

## 6.22 Backpropagation

Once the loss is calculated,

the error is propagated backwards through the network.

```text
Prediction

↓

Loss

↓

Dense Layer

↓

RNN Layer

↓

Embedding Layer
```

During this backward pass,

TensorFlow computes gradients for every trainable parameter.

The embedding vectors also receive gradients.

These gradients indicate how each embedding value should change to reduce the prediction error.

---

## 6.23 Gradient Descent

After gradients are computed,

the optimizer (Adam in this project)

updates every trainable embedding value.

Example

Before Update

```text
good

[0.12, -0.44, 0.76]
```

After One Training Step

```text
good

[0.15, -0.41, 0.79]
```

Only small adjustments are made during each update.

These updates happen repeatedly for every batch during every epoch.

---

## 6.24 How Similar Words Become Similar

Initially,

```text
good

[0.12, -0.44, 0.76]

excellent

[-0.83, 0.51, -0.17]
```

The vectors are completely unrelated.

However,

during training,

the model repeatedly observes sentences such as

```text
This phone is good.

This phone is excellent.

Amazing product.

Excellent quality.

Very good phone.
```

Since the words **good**, **excellent**, and **amazing** frequently appear in similar contexts,

gradient updates gradually move their vectors closer together.

Eventually,

their embeddings become similar.

Similarly,

words such as

```text
bad

terrible

worst

awful
```

also become close to one another in the embedding space.

The model learns these relationships automatically.

No human manually defines them.

---

## 6.25 Complete Learning Process

The complete learning process of the Embedding Layer can be summarized as follows.

```text
Create Embedding Matrix

↓

Initialize Random Values

↓

Input Token IDs

↓

Retrieve Embedding Vectors

↓

RNN Processing

↓

Prediction

↓

Calculate Loss

↓

Backpropagation

↓

Update Embedding Matrix

↓

Repeat for Every Batch

↓

Repeat for Every Epoch
```

With every training iteration,

the embedding vectors become more meaningful,

allowing the model to better understand semantic relationships between words.

---

## 6.26 Important Points

* The Embedding Layer is trainable.
* Embedding vectors are initialized randomly.
* Vectors are updated through backpropagation.
* Similar words gradually obtain similar vector representations.
* The Embedding Layer learns automatically from the training data.
* No manual feature engineering is required.

---

## 6.27 Real Example from This Project

Configuration

```python
Embedding(
    input_dim=5000,
    output_dim=128,
    input_length=100
)
```

For one review

Input

```text
(100)
```

Output

```text
(100 × 128)
```

For one batch

Input

```text
(128 × 100)
```

Output

```text
(128 × 100 × 128)
```

This tensor is directly passed to the RNN layer for sequence learning.
## 6.28 Trainable Embeddings vs Pretrained Embeddings

There are two main approaches for obtaining word embeddings.

### 1. Trainable Embeddings

In this approach, the embedding vectors are learned from scratch during model training.

This is exactly the approach used in this project.

```python
Embedding(
    input_dim=5000,
    output_dim=128
)
```

Initially, the embedding vectors contain random values.

During training, these vectors are updated through backpropagation until they capture meaningful relationships between words.

Advantages

* Learns embeddings specific to the dataset.
* Easy to implement.
* No external embedding files are required.
* Suitable for medium and large datasets.

Disadvantages

* Requires sufficient training data.
* Takes time to learn meaningful embeddings.
* Performance may be lower when the dataset is very small.

---

### 2. Pretrained Embeddings

Instead of learning embeddings from scratch, pretrained embeddings use vectors that have already been trained on extremely large text corpora.

These vectors already understand many relationships between words before the model training begins.

Examples include:

* Word2Vec
* GloVe
* FastText
* BERT Embeddings

Advantages

* Better performance on smaller datasets.
* Already contains semantic knowledge.
* Faster convergence during training.

Disadvantages

* Large embedding files.
* May not perfectly match the project's vocabulary.
* Less flexible for domain-specific language.

---

## 6.29 Common Types of Pretrained Embeddings

### Word2Vec

Word2Vec was developed by Google.

It learns word representations by analyzing the context in which words appear.

Words used in similar contexts receive similar vectors.

Example

```text
King  → Vector

Queen → Similar Vector

Man   → Vector

Woman → Similar Vector
```

Word2Vec is static.

Every occurrence of the word receives the same embedding regardless of sentence context.

---

### GloVe

GloVe (Global Vectors for Word Representation) was developed by Stanford University.

Unlike Word2Vec, GloVe learns embeddings using global word co-occurrence statistics.

It performs well on many NLP tasks and has been widely used in traditional deep learning models.

---

### FastText

FastText was developed by Facebook.

Instead of treating a word as a single unit, FastText breaks each word into smaller character-level components called subwords.

Example

```text
playing

↓

play

lay

aying

ing
```

This allows FastText to generate embeddings even for words that were not seen during training.

---

### BERT Embeddings

BERT uses contextual embeddings.

Unlike Word2Vec or GloVe,

the meaning of a word changes depending on the sentence.

Example

```text
I went to the bank to deposit money.

The fisherman sat on the bank of the river.
```

The word **bank** has different meanings in the two sentences.

BERT generates different embeddings for each occurrence based on context.

This makes BERT much more powerful than traditional embedding techniques.

---

## 6.30 Why Was the Keras Embedding Layer Used in This Project?

The Amazon Review Sentiment Analysis project uses the Keras Embedding Layer instead of pretrained embeddings for the following reasons:

* The objective was to understand how embeddings are learned.
* The dataset is sufficiently large to learn useful embeddings.
* The Embedding layer integrates seamlessly with the RNN/LSTM architecture.
* It simplifies the training pipeline without requiring external embedding files.
* It demonstrates end-to-end deep learning using TensorFlow/Keras.

---

## 6.31 Comparison

| Technique       | Learns During Training                 | Context Aware | Handles Unknown Words | External Files Required |
| --------------- | -------------------------------------- | ------------- | --------------------- | ----------------------- |
| Keras Embedding | Yes                                    | No            | No                    | No                      |
| Word2Vec        | No (Pretrained)                        | No            | No                    | Yes                     |
| GloVe           | No (Pretrained)                        | No            | No                    | Yes                     |
| FastText        | No (Pretrained)                        | No            | Yes                   | Yes                     |
| BERT            | Already Pretrained (Can be Fine-Tuned) | Yes           | Yes                   | Yes                     |

---

## 6.32 Interview Questions

### Basic

**Q1. What is an Embedding Layer?**

**Q2. Why is an Embedding Layer required after tokenization?**

**Q3. Why can't integer token IDs be given directly to an RNN?**

**Q4. What is the difference between token IDs and embedding vectors?**

**Q5. What does `output_dim=128` mean?**

### Intermediate

**Q6. Why is the embedding matrix `5000 × 128`?**

**Q7. Why does one review become `100 × 128`?**

**Q8. Why does one batch become `128 × 100 × 128`?**

**Q9. How does the Embedding Layer learn?**

**Q10. Is the Embedding Layer trainable?**

### Advanced

**Q11. What is the difference between trainable and pretrained embeddings?**

**Q12. When would you choose Word2Vec instead of a trainable embedding?**

**Q13. How is FastText different from Word2Vec?**

**Q14. Why is BERT considered contextual?**

**Q15. Why did you choose the Keras Embedding layer for this project?**

---

## 6.33 Common Mistakes

* Assuming token IDs contain semantic meaning.
* Thinking the embedding vectors are manually defined.
* Confusing vocabulary size with sentence length.
* Believing embeddings are fixed during training.
* Forgetting that the embedding matrix is trainable by default.

---

## 6.34 Best Practices

* Fit the tokenizer only on the training data.
* Choose an embedding dimension appropriate for the problem.
* Use pretrained embeddings when working with small datasets or when prior language knowledge is beneficial.
* Save the trained model so the learned embeddings are preserved.
* Use the same tokenizer during inference to maintain consistency.

---

## 6.35 Final Chapter Summary

The Embedding Layer is the first trainable layer in the deep learning model. It converts integer token IDs into dense vector representations that capture semantic relationships between words.

Unlike token IDs, embedding vectors are learned during training through backpropagation. These learned representations allow the RNN to understand meaningful patterns in text rather than arbitrary numerical identifiers.

In this project, the Keras Embedding layer learns 128-dimensional representations for the top 5000 vocabulary words. The output tensor has the shape `(batch_size, MAX_LEN, 128)` and serves as the input to the RNN layer.



