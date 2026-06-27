# Chapter 7: Simple Recurrent Neural Network (RNN)

## 7.1 Introduction

After the Embedding Layer, every customer review is represented as a sequence of dense vectors.

For example, if

```python
MAX_LEN = 100
output_dim = 128
```

then one review becomes

```text
100 × 128
```

This means:

* 100 tokens
* Each token represented by 128 learned features

The next task is to understand the relationship between these tokens.

This is performed by the **Recurrent Neural Network (RNN)**.

Unlike a traditional Artificial Neural Network (ANN), an RNN processes the sequence one token at a time while remembering information from previous tokens.

This memory allows the model to understand the order and context of words.

---

## 7.2 Why Do We Need an RNN?

Consider the following sentence.

```text
The movie was not good.
```

The word

```text
good
```

normally indicates positive sentiment.

However,

```text
not good
```

indicates negative sentiment.

The meaning changes because of the previous word.

A traditional ANN treats every input independently and cannot naturally remember previous words.

An RNN solves this problem by carrying information from previous time steps through a hidden state.

---

## 7.3 ANN vs RNN

### Artificial Neural Network (ANN)

An ANN assumes that all input features are independent.

Example

```text
Age

Salary

Height

Weight
```

The order of these features does not matter.

Therefore, an ANN is suitable for structured tabular data.

---

### Recurrent Neural Network (RNN)

An RNN is designed for sequential data.

Examples include:

* Sentences
* Speech
* Time series
* Sensor data
* Stock prices

Here, the order of the inputs is important.

The RNN processes one token after another while remembering previous information.

---

## 7.4 RNN in This Project

The model contains the following RNN layer.

```python
SimpleRNN(
    units=64,
    return_sequences=False
)
```

This layer receives the output of the Embedding Layer and processes each token sequentially.

---

## 7.5 Input to the RNN

From the previous chapter,

one review entering the RNN has the shape

```text
100 × 128
```

Meaning

* 100 time steps
* 128 features at every time step

The RNN processes these vectors one by one.

---

## 7.6 What is a Time Step?

A time step refers to one position in the input sequence.

Suppose a review contains

```text
I love this phone
```

After padding,

the review may look like

```text
[12, 25, 7, 84, 0, 0, ...]
```

After the Embedding Layer,

the review becomes

```text
100 × 128
```

The RNN processes the sequence in the following order.

```text
Time Step 1

↓

Token 1

↓

1 × 128

----------------

Time Step 2

↓

Token 2

↓

1 × 128

----------------

Time Step 3

↓

Token 3

↓

1 × 128

----------------

...

----------------

Time Step 100

↓

Token 100

↓

1 × 128
```

Each token corresponds to one time step.

---

## 7.7 What is a Hidden State?

The hidden state is the memory of the RNN.

After processing one token,

the RNN stores useful information in a hidden state.

This hidden state is passed to the next time step.

Therefore,

each new token is processed together with the information learned from previous tokens.

The hidden state allows the RNN to remember context throughout the sentence.

---

## 7.8 Example

Sentence

```text
I love this phone
```

The RNN processes it as

```text
Token 1

↓

Hidden State 1

↓

Token 2 + Hidden State 1

↓

Hidden State 2

↓

Token 3 + Hidden State 2

↓

Hidden State 3

↓

Token 4 + Hidden State 3

↓

Final Hidden State
```

The final hidden state summarizes the information from the entire sentence.

It is this final hidden state that is passed to the Dense layer in this project.

---

## 7.9 Why is Memory Important?

Suppose the sentence is

```text
The phone is not good.
```

When the RNN reaches the word

```text
good
```

it already remembers the previous word

```text
not
```

Therefore,

it can interpret the sentiment more accurately than a model that processes each word independently.

---

## 7.10 Chapter Summary

The RNN processes the embedded review one token at a time. At every time step, it combines the current token with the hidden state from the previous time step, allowing it to capture sequential dependencies within the sentence. The final hidden state summarizes the entire review and becomes the input to the Dense layer for sentiment prediction.
7.11 Internal Working of the RNN Cell

Although RNNs are often illustrated as multiple connected cells, an actual Recurrent Neural Network contains only one RNN cell. This single cell is reused repeatedly for every token in the sequence. At each time step, the cell receives the current token along with the hidden state from the previous time step, processes them using the same set of weights, and generates a new hidden state.

Because the same cell and the same weights are reused throughout the sequence, the model can process sentences of different lengths without increasing the number of trainable parameters.

7.12 Structure of an RNN Cell

An RNN cell receives two inputs:

Current Token Embedding
Previous Hidden State

It produces one output:

New Hidden State

The operation can be represented as follows:

                 Previous Hidden State
                         │
                         ▼
                  ┌─────────────┐
Current Token --->│   RNN Cell  │
 (1 × 128)        └─────────────┘
                         │
                         ▼
                New Hidden State
                     (1 × 64)

The newly generated hidden state becomes the previous hidden state for the next time step.

7.13 Processing the First Token

In this project,

Embedding(MAX_FEATURES, 128, input_length=100)
SimpleRNN(units=64)

produces an input of shape:

100 × 128

This means:

100 time steps (tokens)
128 embedding features for each token

At Time Step 1, the RNN receives:

Current Token
Shape : (1 × 128)

Since there is no previous token, the initial hidden state is automatically initialized to zeros.

Previous Hidden State
Shape : (1 × 64)

[0,0,0,0,........0]

The RNN combines these two inputs and produces the first hidden state.

7.14 Hidden State Update

After processing the first token, the hidden state becomes:

Hidden State 1
Shape : (1 × 64)

When the second token arrives, the RNN no longer starts from zero.

Instead, it receives:

Current Token (1 × 128)

+

Hidden State 1 (1 × 64)

and produces

Hidden State 2 (1 × 64)

This process continues until all 100 tokens have been processed.

7.15 Complete Flow for One Sentence

Suppose the sentence is:

I love this phone

After tokenization and padding:

[12, 25, 7, 84, 0, 0, ...]

After the Embedding Layer:

100 × 128

The RNN processes the sentence sequentially:

Token 1 (1×128)
        │
        ▼
Hidden State 1 (1×64)

        │
        ▼
Token 2 (1×128)
        │
        ▼
Hidden State 2 (1×64)

        │
        ▼
Token 3 (1×128)
        │
        ▼
Hidden State 3 (1×64)

        │
        ▼
...

        │
        ▼

Token 100 (1×128)
        │
        ▼
Final Hidden State (1×64)

The final hidden state contains information learned from the entire sentence. Since return_sequences=False is used in this project, only this final hidden state is passed to the next layer.

## 7.16 Processing a Batch of Sentences

Until now, we have explained the RNN using a **single sentence** because it is easier to understand the concept of hidden states.

However, during actual training, the RNN does **not** process one sentence at a time.

Instead, it processes an entire **batch** of sentences simultaneously.

Suppose:

```python
batch_size = 50
MAX_LEN = 100
embedding_dim = 128
```

After the Embedding Layer, the input to the RNN has the shape:

```text
(50, 100, 128)
```

This means:

* 50 sentences in one batch
* 100 tokens in every sentence
* Each token represented by 128 embedding features

The RNN processes the batch one time step at a time.

At **Time Step 1**, the RNN receives the **first token from all 50 sentences simultaneously**.

Graphically,

```text
Sentence 1   → Token 1
Sentence 2   → Token 1
Sentence 3   → Token 1
...
Sentence 50  → Token 1

↓

Processed Together
```

The RNN produces one hidden state for each sentence.

Therefore, after Time Step 1, the hidden state has the shape:

```text
(50, 64)
```

Meaning:

* One hidden state for every sentence.
* Each hidden state contains 64 learned features.

At **Time Step 2**, the RNN processes the second token from every sentence together.

```text
Sentence 1   → Token 2
Sentence 2   → Token 2
Sentence 3   → Token 2
...
Sentence 50  → Token 2

↓

Processed Together
```

Again, the hidden states are updated for all 50 sentences simultaneously.

This process continues until all 100 time steps have been processed.

---

## 7.17 Complete Batch Processing

The complete execution of one batch can be visualized as:

```text
Input from Embedding Layer

Shape = (50,100,128)

────────────────────────────────────

Time Step 1

↓

50 Token-1 vectors

↓

50 Hidden States

Shape = (50,64)

────────────────────────────────────

Time Step 2

↓

50 Token-2 vectors

↓

50 Updated Hidden States

Shape = (50,64)

────────────────────────────────────

...

────────────────────────────────────

Time Step 100

↓

50 Token-100 vectors

↓

Final Hidden States

Shape = (50,64)
```

Although we often draw an RNN processing one sentence, modern deep learning libraries such as TensorFlow and Keras process **all sentences in the batch in parallel** at each time step.

This makes training much faster while preserving the sequential order **within each sentence**.

For example:

* Sentence 1 still processes Token 1 → Token 2 → Token 3 → ... → Token 100.
* Sentence 2 also processes Token 1 → Token 2 → Token 3 → ... → Token 100.

The important point is that **the same time step across all sentences is computed together**.

### 7.18 Key Takeaways
An RNN contains one reusable cell, not one separate cell per token.
Every token is processed one after another.
Each time step receives the current token embedding and the previous hidden state.
The hidden state acts as the memory of the network.
The size of the hidden state is determined by the units parameter.
The same weights are shared across all time steps.
The final hidden state summarizes the information contained in the entire sentence and is passed to the next layer when return_sequences=False.

## Key Observation

Suppose we have:

```python
batch_size = 128
MAX_LEN = 100
units = 64
```

Then the execution flow becomes:

```text
Embedding Output

(128,100,128)

↓

Time Step 1

Input Shape

(128,128)

↓

Hidden State

(128,64)

↓

Time Step 2

Input Shape

(128,128)

↓

Hidden State

(128,64)

↓

...

↓

Time Step 100

↓

Final Hidden State

(128,64)
```

Notice something very important.

The RNN moves **forward through time**, not through the batch.

The batch dimension is processed **in parallel**, while the time dimension is processed **sequentially**.

This is one of the key ideas behind how RNNs are efficiently implemented in TensorFlow and Keras.

