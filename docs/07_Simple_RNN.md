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

## 7.19 Understanding `return_sequences`

While processing a sentence, the RNN generates a hidden state at every time step.

Suppose a sentence contains **100 tokens**. The RNN produces a hidden state after processing each token.

```
Hidden State 1
      │
      ▼
Hidden State 2
      │
      ▼
Hidden State 3
      │
      ▼
...
      │
      ▼
Hidden State 100
```

Each hidden state contains **64 features** because the RNN in this project is defined as:

```python
SimpleRNN(
    units=64
)
```

The parameter `return_sequences` determines **which hidden states are returned as the output of the RNN layer**.

---

## 7.20 `return_sequences=False`

This is the configuration used in this project.

```python
SimpleRNN(
    units=64,
    return_sequences=False
)
```

When `return_sequences=False`, the RNN returns **only the final hidden state** after processing the complete sentence.

```
Hidden State 1
      │
      ▼
Hidden State 2
      │
      ▼
Hidden State 3
      │
      ▼
...
      │
      ▼
Hidden State 100
      │
      ▼
Returned as Output
```

Since only the last hidden state is returned, the output shape becomes:

```
(batch_size, 64)
```

For example, if:

```python
batch_size = 128
```

the output shape is:

```
(128, 64)
```

This means:

- 128 reviews are processed in one batch.
- Each review produces one final hidden state.
- Every hidden state contains 64 learned features.

The final hidden state summarizes the information learned from the entire sentence and is passed to the next layer.

---

## 7.21 `return_sequences=True`

When the RNN is created as:

```python
SimpleRNN(
    units=64,
    return_sequences=True
)
```

the RNN returns **every hidden state**, not just the final one.

```
Hidden State 1
      │
      ▼
Hidden State 2
      │
      ▼
Hidden State 3
      │
      ▼
...
      │
      ▼
Hidden State 100

↓

Return All Hidden States
```

The output shape becomes:

```
(batch_size, 100, 64)
```

For example, if:

```python
batch_size = 128
```

the output shape becomes:

```
(128, 100, 64)
```

This means:

- 128 reviews in one batch.
- 100 hidden states for every review.
- Each hidden state contains 64 learned features.

Instead of returning one vector per review, the RNN returns the complete sequence of hidden states.

---

## 7.22 Comparison of `return_sequences`

| `return_sequences=False` | `return_sequences=True` |
|---------------------------|-------------------------|
| Returns only the final hidden state | Returns every hidden state |
| Output Shape: `(batch_size, 64)` | Output Shape: `(batch_size, 100, 64)` |
| Produces one feature vector per review | Produces one feature vector for every token |
| Used for text classification | Used when another sequence-processing layer follows |
| Used in this project | Commonly used in stacked RNN/LSTM architectures |

---

## 7.23 Why Was `return_sequences=False` Used in This Project?

The objective of this project is to predict **one sentiment** for the entire customer review.

Although every review contains 100 tokens, the model only needs **one final prediction** (Positive or Negative).

Therefore, only the **final hidden state** is required.

The model architecture is:

```python
Embedding(MAX_FEATURES, 128, input_length=100)

↓

SimpleRNN(64, return_sequences=False)

↓

Dropout(0.5)

↓

Dense(32)

↓

Dense(1)
```

The output of the RNN is:

```
(batch_size, 64)
```

This output becomes the input to the Dense layer.

If `return_sequences=True` had been used, the RNN would return:

```
(batch_size, 100, 64)
```

The Dense layer in this project expects **one feature vector for each review**, not a sequence of 100 hidden states.

Therefore, `return_sequences=False` is the correct choice for this sentiment classification model.

---

## 7.24 When Should `return_sequences=True` Be Used?

`return_sequences=True` is useful when another sequence-processing layer needs access to every hidden state.

Common use cases include:

- Stacked RNNs
- Stacked LSTMs
- Sequence-to-Sequence Models
- Machine Translation
- Named Entity Recognition (NER)
- Speech Recognition

Example:

```python
Embedding(...)

↓

LSTM(128, return_sequences=True)

↓

LSTM(64, return_sequences=False)

↓

Dense(...)
```

The first LSTM returns all hidden states so that the second LSTM can continue processing the sequence.

---

## Key Takeaways

- The RNN generates one hidden state at every time step.
- `return_sequences=False` returns only the final hidden state.
- `return_sequences=True` returns every hidden state.
- The output shape with `return_sequences=False` is `(batch_size, 64)`.
- The output shape with `return_sequences=True` is `(batch_size, 100, 64)`.
- Since this project performs **sentiment classification**, only one prediction is required for each review.
- Therefore, `return_sequences=False` is the appropriate choice for this model.
## 7.25 Complete Data Flow Through the Model

Now that we understand the Embedding Layer and the RNN, let's trace how a single batch of reviews flows through the complete model.

The model architecture is:

```python
Embedding(MAX_FEATURES, 128, input_length=MAX_LEN)

↓

SimpleRNN(64, return_sequences=False)

↓

Dropout(0.5)

↓

Dense(32, activation='relu')

↓

Dense(1, activation='sigmoid')
```

Suppose:

```python
batch_size = 128
MAX_LEN = 100
embedding_dim = 128
units = 64
```

The input and output shapes at each layer are shown below.

```
Customer Reviews
Shape = (128)

↓

Tokenization

↓

Padding

Shape = (128,100)

↓

Embedding Layer

Shape = (128,100,128)

↓

SimpleRNN

Shape = (128,64)

↓

Dropout

Shape = (128,64)

↓

Dense(32)

Shape = (128,32)

↓

Dense(1)

Shape = (128,1)

↓

Prediction
```

Each layer transforms the data into a representation that is more useful for sentiment classification.

---

## 7.26 Why is Dropout Used?

After the RNN processes the sequence, it produces one feature vector of size 64 for every review.

Example:

```
(128,64)
```

These features are highly informative, but the model may begin to memorize patterns from the training data instead of learning general patterns.

This phenomenon is known as **overfitting**.

To reduce overfitting, the following layer is used:

```python
Dropout(0.5)
```

During training, Dropout randomly sets approximately **50% of the neurons to zero** in every batch.

Example:

Before Dropout

```
[0.91, 0.45, 0.76, 0.82, 0.33, 0.64]
```

After Dropout

```
[0.91, 0.00, 0.76, 0.00, 0.33, 0.00]
```

Notice that the **shape does not change**.

Before Dropout

```
(128,64)
```

After Dropout

```
(128,64)
```

Only some neuron values are temporarily ignored during training.

This forces the model to learn more robust features instead of depending on only a few neurons.

During prediction (`model.predict()`), Dropout is automatically disabled, and all neurons are used.

---

## 7.27 Why is Dense(32) Used?

After the RNN, each review is represented by a feature vector of length 64.

```
(128,64)
```

Instead of making the prediction immediately, the model first passes these features through a fully connected layer.

```python
Dense(32, activation='relu')
```

This layer combines and transforms the 64 features into 32 higher-level features.

Graphically,

```
64 Features

↓

Dense Layer

↓

32 Features
```

Every one of the 64 input features is connected to all 32 neurons.

This allows the model to learn more complex combinations of the information extracted by the RNN.

The output shape becomes:

```
(128,32)
```

---

## 7.28 Why Not Directly Use Dense(1)?

A common interview question is:

**"Why didn't you connect the RNN directly to Dense(1)?"**

Technically, you can.

For example:

```python
SimpleRNN(64)

↓

Dense(1)
```

This architecture is valid and often works well for simple problems.

However, adding an intermediate Dense layer provides the model with additional learning capacity.

Instead of immediately making the prediction, the network first learns richer feature combinations.

```
64 Features

↓

Dense(32)

↓

32 Better Features

↓

Dense(1)

↓

Prediction
```

This often improves the model's ability to capture complex relationships within the learned RNN features.

---

## 7.29 Final Prediction Layer

The final layer in the model is:

```python
Dense(1, activation='sigmoid')
```

This layer contains only **one neuron** because the project performs **binary sentiment classification**.

The sigmoid activation function converts the output into a probability between **0 and 1**.

Example:

```
Output = 0.92
```

Interpretation:

- Values close to **1** indicate Positive sentiment.
- Values close to **0** indicate Negative sentiment.

Typically, a threshold of **0.5** is used.

```
Probability ≥ 0.5

↓

Positive Review

Probability < 0.5

↓

Negative Review
```

---

## 7.30 Complete Prediction Pipeline

The complete prediction flow for one review is:

```
Customer Review

↓

Preprocessing

↓

Tokenization

↓

Padding

↓

Embedding

(100,128)

↓

SimpleRNN

(64)

↓

Dropout

(64)

↓

Dense(32)

↓

Dense(1)

↓

Sigmoid

↓

Prediction Probability

↓

Positive / Negative
```

This pipeline transforms raw customer text into a final sentiment prediction through a series of preprocessing and deep learning operations.

---

## Key Takeaways

- The Embedding Layer converts token IDs into dense vectors.
- The RNN learns sequential information from these vectors.
- Dropout reduces overfitting during training.
- Dense(32) learns higher-level feature combinations.
- Dense(1) with sigmoid produces the final sentiment probability.
- Every layer has a specific role in transforming raw text into an accurate sentiment prediction.

## 7.31 Backpropagation Through Time (BPTT)

Until now, we have learned how an RNN processes a sentence in the forward direction.

However, an important question remains:

**How does the RNN learn the correct weights?**

The answer is **Backpropagation Through Time (BPTT)**.

BPTT is an extension of the backpropagation algorithm used in Artificial Neural Networks. Since an RNN processes sequential data over multiple time steps, the error must also be propagated backward through all of those time steps.

---

## 7.32 Forward Pass

Suppose the input review is:

```
I love this phone
```

After preprocessing, tokenization, padding, and embedding, the RNN receives:

```
100 × 128
```

The RNN processes the review one token at a time.

```
Token 1
      │
      ▼
Hidden State 1
      │
      ▼
Token 2
      │
      ▼
Hidden State 2
      │
      ▼
...
      │
      ▼
Token 100
      │
      ▼
Final Hidden State
      │
      ▼
Dense(32)
      │
      ▼
Dense(1)
      │
      ▼
Prediction
```

Suppose the model predicts:

```
Prediction = Positive

Probability = 0.62
```

The actual label is:

```
Negative

Label = 0
```

Since the prediction is incorrect, the model must adjust its parameters.

---

## 7.33 Loss Calculation

The prediction is compared with the actual label using the Binary Crossentropy loss function.

```
Prediction

↓

Actual Label

↓

Binary Crossentropy Loss
```

A higher loss indicates that the prediction is far from the correct answer.

The objective of training is to reduce this loss.

---

## 7.34 Backward Pass

Once the loss is calculated, TensorFlow begins the backward pass.

The error flows through the network in the reverse direction.

```
Prediction

↓

Dense(1)

↓

Dense(32)

↓

SimpleRNN

↓

Embedding Layer
```

Unlike an ANN, the RNN must also update the hidden states that were generated at every time step.

Therefore, the error is propagated backward through the sequence.

```
Hidden State100

↓

Hidden State99

↓

Hidden State98

↓

...

↓

Hidden State2

↓

Hidden State1
```

This reverse flow through all time steps is called **Backpropagation Through Time (BPTT)**.

---

## 7.35 Weight Updates

During backpropagation, TensorFlow computes gradients for every trainable parameter.

These include:

- Embedding Layer weights
- RNN weights
- Dense layer weights

The optimizer (Adam in this project) uses these gradients to update the parameters.

```
Old Weights

↓

Compute Gradients

↓

Adam Optimizer

↓

Updated Weights
```

These updates help the model make more accurate predictions in future training iterations.

---

## 7.36 Why Are All Time Steps Updated?

One of the unique properties of an RNN is that every hidden state contributes to the final prediction.

Therefore, if the final prediction is incorrect, the model cannot simply update the last hidden state.

Instead, it must update the computations performed at every previous time step because each hidden state influenced the next one.

```
Token 1

↓

Hidden State1

↓

Hidden State2

↓

Hidden State3

↓

...

↓

Hidden State100

↓

Prediction
```

If Hidden State 1 contains incorrect information, that error propagates through the entire sequence.

Therefore, BPTT updates the model across all time steps.

---

## 7.37 Difference Between Backpropagation and BPTT

| Backpropagation (ANN) | Backpropagation Through Time (RNN) |
|------------------------|-------------------------------------|
| Used in Feedforward Networks | Used in Recurrent Neural Networks |
| Error flows backward through layers | Error flows backward through layers **and time steps** |
| No memory between inputs | Hidden states connect time steps |
| Simpler computation | More computationally intensive |

---

## 7.38 Vanishing Gradient Problem

Although RNNs can theoretically remember information from many previous time steps, they often struggle with very long sequences.

During BPTT, gradients become smaller as they travel backward through many time steps.

Eventually, the gradients may become so small that earlier time steps receive almost no updates.

This phenomenon is known as the **Vanishing Gradient Problem**.

As a result, Simple RNNs tend to remember only recent information and may forget important information from earlier in the sequence.

This limitation led to the development of **Long Short-Term Memory (LSTM)** networks, which are specifically designed to preserve long-term information.

The next chapter explains how LSTMs solve this problem.

---

## Key Takeaways

- RNNs learn using **Backpropagation Through Time (BPTT)**.
- BPTT is an extension of standard backpropagation for sequential data.
- The prediction error flows backward through both the network layers and the sequence of time steps.
- All trainable parameters, including the Embedding Layer, RNN, and Dense layers, are updated during training.
- Long sequences can cause the **Vanishing Gradient Problem**, making it difficult for Simple RNNs to remember information from the beginning of a sequence.
- LSTM networks were introduced to overcome this limitation.
## 7.39 Interview Questions

The following are common interview questions related to the RNN architecture used in this project.

### Basic Questions

### Q1. What is a Recurrent Neural Network (RNN)?

**Answer:**

A Recurrent Neural Network (RNN) is a type of neural network designed to process sequential data. Unlike a traditional Artificial Neural Network, an RNN maintains a hidden state (memory) that carries information from previous time steps, allowing it to learn relationships between words in a sequence.

---

### Q2. Why did you choose an RNN for this project?

**Answer:**

This project performs sentiment analysis on customer reviews. Since the order of words affects the meaning of a sentence, an RNN is more suitable than a traditional ANN because it processes words sequentially and remembers previous information using hidden states.

---

### Q3. What was the input to your RNN?

**Answer:**

The input to the RNN was the output of the Embedding Layer.

Input Shape:

```
(batch_size,100,128)
```

where:

- batch_size = Number of reviews processed together.
- 100 = Number of tokens in each padded review.
- 128 = Embedding features for each token.

---

### Q4. What does `units=64` mean?

**Answer:**

The parameter `units=64` specifies the number of neurons inside the RNN.

Therefore, every hidden state contains 64 learned features.

The output shape becomes:

```
(batch_size,64)
```

when `return_sequences=False`.

---

### Q5. What is a hidden state?

**Answer:**

A hidden state is the internal memory of an RNN.

It stores information learned from previous tokens and passes that information to the next time step, enabling the model to understand context within a sentence.

---

### Intermediate Questions

### Q6. How does an RNN process a sentence?

**Answer:**

The RNN processes one token at a time.

At every time step, it receives:

- Current token embedding
- Previous hidden state

It combines both to generate a new hidden state.

This continues until the final token has been processed.

---

### Q7. Does the RNN create 100 different cells for a sentence of length 100?

**Answer:**

No.

The RNN contains only one cell.

The same cell is reused for every token.

The same trainable weights are shared across all time steps.

---

### Q8. What is weight sharing?

**Answer:**

Weight sharing means the RNN uses the same trainable parameters at every time step instead of creating a new network for each token.

This keeps the number of parameters constant regardless of sentence length.

---

### Q9. Explain batch processing in an RNN.

**Answer:**

During training, the RNN processes an entire batch of reviews simultaneously.

For example:

```
Embedding Output

(128,100,128)
```

At Time Step 1, the first token from all 128 reviews is processed together.

The hidden state becomes:

```
(128,64)
```

The process continues for Time Steps 2, 3, ..., 100.

The batch dimension is processed in parallel, while the time dimension is processed sequentially.

---

### Q10. What is `return_sequences=False`?

**Answer:**

It returns only the final hidden state.

Output Shape:

```
(batch_size,64)
```

This is suitable for classification tasks where one prediction is required for the entire sequence.

---

### Q11. What is `return_sequences=True`?

**Answer:**

It returns the hidden state from every time step.

Output Shape:

```
(batch_size,100,64)
```

This is commonly used when another RNN/LSTM layer follows.

---

### Q12. Why did you use `return_sequences=False`?

**Answer:**

This project predicts one sentiment label for the entire review.

Only the final hidden state is required because it summarizes the complete sentence.

Therefore, `return_sequences=False` was used.

---

### Q13. Why is Dropout used after the RNN?

**Answer:**

Dropout reduces overfitting by randomly disabling a fraction of neurons during training.

This encourages the model to learn more general patterns instead of memorizing the training data.

---

### Q14. Why did you use `Dense(32)` before `Dense(1)`?

**Answer:**

The intermediate Dense layer learns higher-level feature combinations from the 64 RNN features before making the final prediction.

Although `Dense(1)` can be connected directly to the RNN, using `Dense(32)` often improves learning capacity.

---

### Advanced Questions

### Q15. What is Backpropagation Through Time (BPTT)?

**Answer:**

Backpropagation Through Time is the learning algorithm used by RNNs.

It propagates the prediction error backward through both the network layers and the sequence of time steps to update all trainable parameters.

---

### Q16. What is the Vanishing Gradient Problem?

**Answer:**

During Backpropagation Through Time, gradients become progressively smaller as they travel backward through many time steps.

As a result, the RNN struggles to learn long-term dependencies because earlier time steps receive very small updates.

---

### Q17. How does LSTM solve this problem?

**Answer:**

LSTM introduces a memory cell and gating mechanisms (Forget Gate, Input Gate, and Output Gate) that preserve important information over long sequences.

This significantly reduces the vanishing gradient problem.

---

### Q18. What are the limitations of a Simple RNN?

**Answer:**

- Suffers from vanishing gradients.
- Cannot effectively remember long-term dependencies.
- Performance decreases for long sequences.
- Slower than feedforward networks due to sequential processing.

---

### Q19. Can an RNN process sentences of different lengths?

**Answer:**

Yes.

However, during batch training, all sequences are padded to the same length using `pad_sequences()` so that they can be processed efficiently in parallel.

---

### Q20. Explain your complete RNN model architecture.

**Answer:**

```
Customer Review

↓

Preprocessing

↓

Tokenization

↓

Padding

↓

Embedding Layer

↓

SimpleRNN(64)

↓

Dropout(0.5)

↓

Dense(32)

↓

Dense(1)

↓

Sigmoid

↓

Positive / Negative
```

This pipeline converts raw customer reviews into sentiment predictions.

---

## 7.40 Common Mistakes

- Confusing hidden states with output vectors.
- Assuming an RNN creates a different cell for every token.
- Thinking token IDs can be directly understood by the RNN.
- Forgetting that all time steps share the same weights.
- Using `return_sequences=True` when only one prediction is required.
- Ignoring the vanishing gradient problem for long sequences.

---

## 7.41 Best Practices

- Use sequence padding before feeding data into an RNN.
- Choose the number of units based on the complexity of the problem.
- Apply Dropout to reduce overfitting.
- Use `return_sequences=False` for sequence classification tasks.
- Save both the trained model and tokenizer for inference.
- Consider using LSTM or GRU for longer sequences.

---

## 7.42 Chapter Summary

In this chapter, we explored the complete working of a Simple Recurrent Neural Network (RNN).

We learned how the RNN processes sequential data one token at a time, how hidden states carry information through the sequence, how batches are processed efficiently, and how Backpropagation Through Time updates the model during training.

We also understood why `return_sequences=False` was appropriate for this sentiment classification project, how Dropout and Dense layers contribute to better generalization, and why Simple RNNs struggle with long-term dependencies.

The next chapter introduces **Long Short-Term Memory (LSTM)** networks, which extend the RNN architecture by introducing memory cells and gating mechanisms to overcome the vanishing gradient problem and better capture long-term dependencies.

