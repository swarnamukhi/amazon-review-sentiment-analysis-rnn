# Chapter 8: Long Short-Term Memory (LSTM)

## 8.1 Introduction

In the previous chapter, we learned how a Simple Recurrent Neural Network (RNN) processes sequential data by maintaining a hidden state.

Although RNNs can remember information from previous time steps, they struggle to retain information over long sequences due to the **Vanishing Gradient Problem**.

To overcome this limitation, Long Short-Term Memory (LSTM) networks were introduced.

LSTM is a specialized type of Recurrent Neural Network that can selectively remember important information and forget irrelevant information over long sequences.

Because of this capability, LSTMs generally perform better than Simple RNNs for many Natural Language Processing (NLP) tasks.

---

## 8.2 Why Was LSTM Introduced?

Consider the following sentence:

```

I grew up in Hyderabad.

After completing my engineering, I worked in several companies.

Many years later I moved to Ladakh.

Today I still remember Hyderabad.

```

To correctly understand the last sentence, the model must remember the word **Hyderabad**, which appeared many words earlier.

A Simple RNN often fails to retain such long-term information because the hidden state gradually loses earlier information during training.

This limitation is called the **Vanishing Gradient Problem**.

LSTM was specifically designed to preserve important information over long sequences.

---

## 8.3 Limitation of Simple RNN

The hidden state of an RNN is updated repeatedly.

```

Hidden State 1

↓

Hidden State 2

↓

Hidden State 3

↓

...

↓

Hidden State 100

```

As the sequence becomes longer, information from the beginning gradually fades.

Consequently, the model focuses mainly on recent words and may ignore important words that appeared much earlier.

---

## 8.4 How Does LSTM Solve This Problem?

Instead of relying only on a hidden state, an LSTM introduces a second memory called the **Cell State**.

The Cell State acts like a long-term memory that flows through the entire sequence.

Instead of replacing all previous information, the LSTM learns:

- What information should be remembered.
- What information should be forgotten.
- What new information should be stored.
- What information should be passed to the next layer.

This selective memory mechanism enables the LSTM to capture long-term dependencies much more effectively than a Simple RNN.

---

## 8.5 RNN vs LSTM

| Simple RNN | LSTM |
|------------|------|
| Maintains only Hidden State | Maintains Hidden State and Cell State |
| Difficult to remember long sequences | Can remember long-term information |
| Suffers from Vanishing Gradient | Greatly reduces the Vanishing Gradient Problem |
| Simpler architecture | More complex architecture with gates |
| Faster training | Slightly slower due to additional computations |
| Suitable for shorter sequences | Suitable for longer sequences |

---

## 8.6 LSTM in This Project

The LSTM model used in this project is:

```python
Embedding(MAX_FEATURES,128,input_length=MAX_LEN)

↓

LSTM(64,return_sequences=False)

↓

Dropout(0.5)

↓

Dense(32,activation='relu')

↓

Dense(1,activation='sigmoid')
```

Notice that the only architectural difference from the Simple RNN model is that the `SimpleRNN` layer is replaced by an `LSTM` layer.

The remaining architecture remains unchanged.

---

## 8.7 Input to the LSTM

The Embedding Layer produces the following output:

```

(batch_size,100,128)

```

Meaning:

- Batch of reviews
- 100 tokens per review
- 128 embedding features per token

This tensor becomes the input to the LSTM layer.

---

## 8.8 Output of the LSTM

The LSTM processes the input sequence one time step at a time.

When

```python
LSTM(
    units=64,
    return_sequences=False
)
```

is used, the output shape becomes:

```

(batch_size,64)

```

The output is then passed to:

```

Dropout

↓

Dense(32)

↓

Dense(1)

```

exactly like the Simple RNN model.

---

## 8.9 Key Takeaways

- LSTM is an improved version of a Simple RNN.
- It introduces a Cell State to preserve long-term information.
- LSTM uses gates to decide what information should be remembered or forgotten.
- It significantly reduces the Vanishing Gradient Problem.
- The output shape is identical to a Simple RNN when the same number of units is used.
## 8.10 Internal Architecture of an LSTM Cell

Although an LSTM processes a sequence like a Simple RNN, its internal architecture is much more sophisticated.

A Simple RNN maintains only a **Hidden State**, whereas an LSTM maintains two types of memory:

1. **Hidden State (hₜ)** – Short-term memory that is passed to the next time step and also used as the output of the current time step.

2. **Cell State (Cₜ)** – Long-term memory that carries important information throughout the sequence.

Unlike a Simple RNN, an LSTM does not blindly overwrite its memory at every time step. Instead, it carefully decides:

- What information should be forgotten.
- What new information should be stored.
- What information should be passed to the next time step.

This decision-making process is performed using three gates.

---

## 8.11 Components of an LSTM Cell

An LSTM cell consists of the following main components:

- Previous Hidden State (hₜ₋₁)
- Previous Cell State (Cₜ₋₁)
- Current Input (xₜ)
- Forget Gate
- Input Gate
- Candidate Cell State
- Output Gate
- New Cell State (Cₜ)
- New Hidden State (hₜ)

These components work together to determine what information should be retained and what should be discarded.

---

## 8.12 Inputs to an LSTM Cell

At every time step, the LSTM receives three inputs:

### 1. Current Input (xₜ)

This is the embedding vector of the current token.

For our project:

```
Shape = (1 × 128)
```

---

### 2. Previous Hidden State (hₜ₋₁)

This contains the short-term information learned up to the previous time step.

Since our LSTM has:

```python
LSTM(units=64)
```

the hidden state has the shape:

```
(1 × 64)
```

---

### 3. Previous Cell State (Cₜ₋₁)

The Cell State stores long-term information that has been carried forward from previous time steps.

Its shape is also:

```
(1 × 64)
```

Therefore, every LSTM cell receives:

```
Current Input

(1 × 128)

+

Previous Hidden State

(1 × 64)

+

Previous Cell State

(1 × 64)
```

---

## 8.13 Outputs of an LSTM Cell

After processing these inputs, the LSTM produces two outputs:

### 1. New Cell State (Cₜ)

This is the updated long-term memory.

It carries important information forward to the next time step.

Shape:

```
(1 × 64)
```

---

### 2. New Hidden State (hₜ)

This represents the short-term memory after processing the current token.

It is passed to:

- The next LSTM cell.
- The next layer in the network (when required).

Shape:

```
(1 × 64)
```

Thus, unlike a Simple RNN, an LSTM produces **two outputs** at every time step instead of one.

---

## 8.14 Visual Representation of an LSTM Cell

The flow of information inside one LSTM cell can be represented as:

```
                 Previous Hidden State (hₜ₋₁)
                           │
                           │
                           ▼
                     ┌─────────────┐
Current Input ─────► │             │
 (1 × 128)           │    LSTM     │
                     │    Cell     │
Previous Cell State ─►             │
   (Cₜ₋₁)            └─────────────┘
                           │
                  ┌────────┴────────┐
                  ▼                 ▼
          New Cell State      New Hidden State
              (Cₜ)                 (hₜ)
             (1 × 64)            (1 × 64)
```

Notice that the Cell State passes through the entire sequence, allowing important information to be preserved over long distances.

---

## 8.15 Why Does LSTM Need Two Memories?

This is one of the most common interview questions.

A Simple RNN has only one hidden state.

As the sequence becomes longer, the hidden state is continuously updated, causing older information to fade.

LSTM solves this problem by separating memory into:

### Hidden State

- Stores short-term information.
- Used to generate the output at the current time step.
- Continuously updated.

### Cell State

- Stores long-term information.
- Travels almost unchanged through the sequence.
- Updated only when necessary using the gates.

This separation allows the network to preserve important information over much longer sequences.

---

## 8.16 Key Takeaways

- An LSTM cell maintains two memories: Hidden State and Cell State.
- The Hidden State captures short-term information.
- The Cell State captures long-term information.
- Every LSTM cell receives the current input, previous hidden state, and previous cell state.
- Every LSTM cell produces a new hidden state and a new cell state.
- The gates control how information flows through these memories.
## 8.17 Why Does an LSTM Need Gates?

Suppose we are reading the following sentence:

```
The movie was not good.
```

When the LSTM reaches the word:

```
good
```

it should remember that the previous word was

```
not
```

Otherwise, it may incorrectly predict that the review is positive.

Now consider another sentence:

```
The movie was released in 2022.
```

When predicting sentiment, the year

```
2022
```

is usually not important.

Ideally, the model should ignore such unnecessary information.

Therefore, an LSTM needs a mechanism that can answer three important questions at every time step:

1. Should I forget some old information?
2. Should I store the current information?
3. What information should I pass to the next time step?

These decisions are made using three gates.

---

## 8.18 The Three Gates of an LSTM

An LSTM contains three gates.

```
Current Input

↓

Forget Gate

↓

Input Gate

↓

Output Gate

↓

Next Time Step
```

Each gate has a different responsibility.

| Gate | Responsibility |
|------|----------------|
| Forget Gate | Removes unnecessary information |
| Input Gate | Stores important new information |
| Output Gate | Produces the hidden state for the next time step |

Together, these gates allow the LSTM to selectively remember useful information while discarding irrelevant details.

---

## 8.19 Forget Gate

The **Forget Gate** is the first gate inside an LSTM cell.

Its purpose is to decide:

> **"What information from the previous Cell State should I keep, and what should I discard?"**

Imagine reading a long novel.

Not every sentence is equally important.

Some details become irrelevant as the story progresses.

Similarly, while processing text, the LSTM learns to forget information that is no longer useful.

The Forget Gate receives:

- Current Input (xₜ)
- Previous Hidden State (hₜ₋₁)

Using these two inputs, it calculates a value between **0 and 1** for every feature in the Cell State.

```
0  → Forget Completely

0.5 → Keep Some Information

1  → Keep Completely
```

The Forget Gate does **not** make one decision for the entire sentence.

Instead, it makes one decision for **every feature** in the Cell State.

For example, if the Cell State has 64 features:

```
Cell State

(1 × 64)
```

the Forget Gate also produces:

```
Forget Gate Output

(1 × 64)
```

Each value controls one feature independently.

---

## 8.20 Example of the Forget Gate

Suppose the previous Cell State contains the following values:

```
[0.82, 0.41, 0.93, 0.25]
```

The Forget Gate produces:

```
[1.0, 0.0, 0.7, 0.2]
```

The updated Cell State becomes:

```
[0.82, 0.00, 0.65, 0.05]
```

Notice what happened:

- The first feature was kept completely.
- The second feature was removed.
- The third feature was mostly retained.
- The fourth feature was partially retained.

This selective forgetting allows the LSTM to remove unnecessary information while preserving important knowledge.

---

## 8.21 Why is the Forget Gate Important?

Suppose we have the sentence:

```
I bought this phone in 2018.
```

Later in the review:

```
The battery life is excellent.
```

The purchase year may not be useful for predicting sentiment.

The Forget Gate can gradually reduce the importance of information related to the year while preserving information about the battery and overall opinion.

This prevents the Cell State from becoming filled with irrelevant details.

---

## 8.22 Key Points About the Forget Gate

- It is the first gate inside the LSTM.
- It decides what information should remain in the Cell State.
- It produces one value for every Cell State feature.
- Its output values always lie between 0 and 1.
- Values close to 1 preserve information.
- Values close to 0 remove information.
- It helps the LSTM maintain useful long-term memory while discarding irrelevant details.
## 8.23 Input Gate

After the Forget Gate has removed unnecessary information, the next question is:

> **"What new information should be added to the Cell State?"**

This decision is made by the **Input Gate**.

The Input Gate examines the current token and the previous hidden state to determine how much of the current information is important enough to store in the Cell State.

Unlike the Forget Gate, which removes information, the Input Gate is responsible for learning and storing new knowledge.

---

## 8.24 Inputs to the Input Gate

The Input Gate receives the following inputs:

- Current Input (xₜ)
- Previous Hidden State (hₜ₋₁)

For our project:

Current Input

```
(1 × 128)
```

Previous Hidden State

```
(1 × 64)
```

Using these inputs, the Input Gate generates one value between **0 and 1** for every feature in the Cell State.

```
0  → Do not store this information.

0.5 → Store part of the information.

1  → Store completely.
```

If our LSTM contains 64 units,

```
LSTM(units=64)
```

then the Input Gate produces

```
Input Gate Output

(1 × 64)
```

Each value controls one feature of the Cell State independently.

---

## 8.25 Candidate Cell State

The Input Gate alone cannot update the Cell State.

It must first determine **what new information is available**.

This information is called the **Candidate Cell State**.

Think of the Candidate Cell State as a proposal.

It contains all the new information that **could** be stored in memory.

However, not all of this information will actually be stored.

The Input Gate decides how much of the candidate information should be accepted.

Graphically,

```
Current Input

↓

Candidate Cell State

(New Information)

↓

Input Gate decides

↓

Updated Cell State
```

---

## 8.26 Example of the Input Gate

Suppose the Candidate Cell State contains:

```
[0.91, 0.42, 0.76, 0.15]
```

The Input Gate produces:

```
[1.0, 0.2, 0.8, 0.0]
```

The new information added to the Cell State becomes:

```
[0.91, 0.08, 0.61, 0.00]
```

Notice what happened:

- The first feature was fully accepted.
- The second feature was only partially stored.
- The third feature was mostly stored.
- The fourth feature was ignored.

The Input Gate therefore controls how much new information enters the long-term memory.

---

## 8.27 Real-Life Example

Consider the sentence:

```
This phone has an amazing camera.
```

When the LSTM reaches the word:

```
amazing
```

the Input Gate recognizes that this word carries strong sentiment.

Therefore, it allows this information to enter the Cell State.

Now consider another sentence:

```
This phone was launched in 2020.
```

The word:

```
2020
```

may not be useful for sentiment prediction.

The Input Gate may decide to store very little or none of this information.

Thus, the Cell State focuses on important information while ignoring less relevant details.

---

## 8.28 Forget Gate and Input Gate Together

The Forget Gate and Input Gate work as a team.

The Forget Gate removes unnecessary information.

The Input Gate adds useful new information.

```
Old Cell State

↓

Forget Gate

(Remove unnecessary information)

↓

Input Gate

(Add useful information)

↓

Updated Cell State
```

This continuous process allows the Cell State to remain meaningful throughout the sequence.

---

## 8.29 Key Points About the Input Gate

- The Input Gate determines what new information should be stored.
- It receives the current input and previous hidden state.
- It works together with the Candidate Cell State.
- It produces one value between 0 and 1 for every Cell State feature.
- Values close to 1 store more information.
- Values close to 0 reject unnecessary information.
- Together with the Forget Gate, it updates the Cell State intelligently.
## 8.30 Output Gate

After the Cell State has been updated by the Forget Gate and Input Gate, the LSTM must decide:

> **"What information should I expose as the Hidden State?"**

This decision is made by the **Output Gate**.

The Hidden State is the short-term memory of the LSTM. It is passed to:

- The next LSTM cell.
- The next layer in the neural network (if required).

Unlike the Cell State, which stores long-term information, the Hidden State represents only the information that is useful at the current time step.

---

## 8.31 Inputs to the Output Gate

The Output Gate receives:

- Current Input (xₜ)
- Previous Hidden State (hₜ₋₁)
- Updated Cell State (Cₜ)

For our project:

Current Input

```
(1 × 128)
```

Previous Hidden State

```
(1 × 64)
```

Updated Cell State

```
(1 × 64)
```

Using these inputs, the Output Gate calculates one value between **0 and 1** for every feature.

If the LSTM contains 64 units,

```python
LSTM(units=64)
```

the Output Gate produces:

```
(1 × 64)
```

Each value determines how much information from the Cell State should become part of the Hidden State.

---

## 8.32 Example of the Output Gate

Suppose the updated Cell State is:

```
[0.82, 0.45, 0.93, 0.61]
```

The Output Gate produces:

```
[1.0, 0.3, 0.8, 0.0]
```

The resulting Hidden State becomes:

```
[0.82, 0.14, 0.74, 0.00]
```

Notice that:

- The first feature is fully passed.
- The second feature is partially passed.
- The third feature is mostly passed.
- The fourth feature is completely hidden.

Therefore, the Hidden State contains only the information that the Output Gate considers important.

---

## 8.33 Real-Life Example

Consider the sentence:

```
The movie was not good.
```

After processing the word:

```
not
```

the Cell State already contains important information indicating negation.

When the word:

```
good
```

is processed, the Output Gate allows this contextual information to influence the Hidden State.

As a result, the Hidden State represents the phrase:

```
not good
```

rather than simply:

```
good
```

This enables the LSTM to produce a more accurate sentiment prediction.

---

## 8.34 Complete Working of an LSTM Cell

The complete sequence of operations inside one LSTM cell is:

```
Current Input (xₜ)

+

Previous Hidden State (hₜ₋₁)

+

Previous Cell State (Cₜ₋₁)

↓

Forget Gate

↓

Updated Cell State

↓

Input Gate

↓

Candidate Cell State

↓

New Cell State (Cₜ)

↓

Output Gate

↓

New Hidden State (hₜ)
```

Each gate performs a specific task:

- Forget Gate removes unnecessary information.
- Input Gate stores useful new information.
- Candidate Cell State proposes new memory.
- Output Gate generates the Hidden State.

---

## 8.35 One Time Step in an LSTM

Suppose one token enters the LSTM.

```
Embedding Vector

(1 × 128)
```

The LSTM performs the following operations:

```
Step 1

Receive Current Input

↓

Step 2

Read Previous Hidden State

↓

Step 3

Read Previous Cell State

↓

Step 4

Forget Gate removes unnecessary information

↓

Step 5

Input Gate stores useful information

↓

Step 6

Update Cell State

↓

Step 7

Output Gate generates Hidden State

↓

Pass both memories to the next time step
```

This entire sequence is repeated for every token in the sentence.

---

## 8.36 Difference Between Cell State and Hidden State

This is one of the most common interview questions.

| Cell State | Hidden State |
|------------|--------------|
| Long-term memory | Short-term memory |
| Carries information across long sequences | Represents the current output |
| Updated using Forget and Input Gates | Generated using the Output Gate |
| Helps preserve important information | Passed to the next layer and next time step |
| Shape: (1 × 64) | Shape: (1 × 64) |

Although both have the same shape, they serve different purposes.

---

## 8.37 Key Takeaways

- The Output Gate determines what information should become the Hidden State.
- The Hidden State is passed to the next time step and can also be used as the output of the current time step.
- The Cell State stores long-term information.
- The Hidden State stores short-term information.
- The Forget Gate, Input Gate, and Output Gate work together to update the LSTM's memory intelligently.
- This gated architecture enables LSTMs to remember important information over much longer sequences than a Simple RNN.
## 8.38 Batch Processing in an LSTM

Until now, we have explained the working of an LSTM using a single sentence because it is easier to understand the concepts of the Cell State and Hidden State.

However, during actual training, TensorFlow and Keras do not process one sentence at a time.

Instead, they process an entire **batch** of sentences simultaneously.

Suppose our project uses:

```python
batch_size = 128
MAX_LEN = 100
embedding_dim = 128
```

After the Embedding Layer, the input to the LSTM has the shape:

```
(128,100,128)
```

This means:

- 128 customer reviews
- 100 tokens in each review
- 128 embedding features for every token

The LSTM processes one **time step** at a time.

At **Time Step 1**, it receives the first token from all 128 reviews together.

```
Sentence 1  → Token 1

Sentence 2  → Token 1

Sentence 3  → Token 1

...

Sentence128 → Token 1

↓

Processed Together
```

The LSTM generates:

```
Hidden State

(128,64)

Cell State

(128,64)
```

These states are then used when processing the second token.

---

## 8.39 Processing Through Time

The complete execution of one batch can be visualized as:

```
Embedding Output

Shape = (128,100,128)

──────────────────────────────

Time Step 1

↓

128 Token-1 vectors

↓

Hidden State (128,64)

↓

Cell State (128,64)

──────────────────────────────

Time Step 2

↓

128 Token-2 vectors

↓

Updated Hidden State

↓

Updated Cell State

──────────────────────────────

...

──────────────────────────────

Time Step 100

↓

128 Token-100 vectors

↓

Final Hidden State

(128,64)

↓

Final Cell State

(128,64)
```

Notice something important.

The batch dimension is processed **in parallel**, while the time dimension is processed **sequentially**.

This is exactly the same concept that we learned in the RNN chapter.

---

## 8.40 `return_sequences=False` in LSTM

The LSTM used in this project is:

```python
LSTM(
    units=64,
    return_sequences=False
)
```

The LSTM generates one Hidden State at every time step.

```
Hidden State 1

↓

Hidden State 2

↓

Hidden State 3

↓

...

↓

Hidden State 100
```

When `return_sequences=False` is used, only the **final Hidden State** is returned.

Output Shape:

```
(batch_size,64)
```

For our project:

```
(128,64)
```

This final Hidden State summarizes the information learned from the entire customer review.

---

## 8.41 `return_sequences=True` in LSTM

When

```python
LSTM(
    units=64,
    return_sequences=True
)
```

is used,

the LSTM returns every Hidden State.

Output Shape:

```
(batch_size,100,64)
```

For our project:

```
(128,100,64)
```

This output is typically used when another LSTM layer follows.

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

## 8.42 Why Did We Use `return_sequences=False`?

The objective of this project is to predict **one sentiment** for the entire review.

Therefore, the model only needs one feature vector that summarizes the complete review.

The final Hidden State serves this purpose.

The architecture is:

```python
Embedding(MAX_FEATURES,128,input_length=100)

↓

LSTM(64, return_sequences=False)

↓

Dropout(0.5)

↓

Dense(32)

↓

Dense(1)
```

The output of the LSTM is:

```
(128,64)
```

which becomes the input to the Dense layer.

If `return_sequences=True` had been used,

the output would have been:

```
(128,100,64)
```

Since the Dense layer in this architecture expects one feature vector for each review, `return_sequences=False` is the correct choice.

---

## 8.43 Complete Data Flow Through Our LSTM Model

The complete flow of data through our sentiment analysis model is shown below.

```
Customer Reviews

↓

Preprocessing

↓

Tokenization

↓

Padding

↓

(128,100)

↓

Embedding Layer

↓

(128,100,128)

↓

LSTM Layer

↓

(128,64)

↓

Dropout

↓

(128,64)

↓

Dense(32)

↓

(128,32)

↓

Dense(1)

↓

(128,1)

↓

Sigmoid

↓

Positive / Negative
```

This pipeline converts raw customer reviews into sentiment predictions through a sequence of preprocessing and deep learning operations.

---

## 8.44 Key Takeaways

- LSTM processes batches in parallel and time steps sequentially.
- At every time step, both the Hidden State and Cell State are updated.
- `return_sequences=False` returns only the final Hidden State.
- `return_sequences=True` returns the Hidden State from every time step.
- Our project predicts one sentiment for each review, so `return_sequences=False` is the correct choice.
- The final Hidden State is passed to the Dense layers for classification.
## 8.45 Backpropagation in LSTM

Until now, we have learned how an LSTM performs the forward pass using the Forget Gate, Input Gate, Cell State, and Output Gate.

However, an important question remains:

**How does the LSTM learn the correct gate values and memory representations?**

The answer is **Backpropagation Through Time (BPTT)**.

Like a Simple RNN, an LSTM processes the sequence in the forward direction and computes a prediction. During training, the prediction error is propagated backward through every time step to update all trainable parameters.

---

## 8.46 Forward Pass

Suppose the input review is:

```
This phone is not good.
```

After preprocessing, tokenization, padding, and embedding, the input to the LSTM becomes:

```
(100,128)
```

The LSTM processes one token at a time.

For every token, the following operations occur:

```
Current Token

↓

Forget Gate

↓

Input Gate

↓

Candidate Cell State

↓

Update Cell State

↓

Output Gate

↓

Hidden State
```

After processing all 100 tokens, the final Hidden State is passed to:

```
Dropout

↓

Dense(32)

↓

Dense(1)

↓

Sigmoid

↓

Prediction
```

Suppose the model predicts:

```
Positive

Probability = 0.72
```

but the correct label is:

```
Negative

Label = 0
```

The prediction error must now be used to improve the model.

---

## 8.47 Loss Calculation

The predicted probability is compared with the true label using the Binary Crossentropy loss function.

```
Prediction

↓

Actual Label

↓

Binary Crossentropy Loss
```

A higher loss indicates that the prediction is further from the correct answer.

The objective of training is to minimize this loss.

---

## 8.48 Backward Pass

After the loss is calculated, TensorFlow performs Backpropagation Through Time.

The error flows backward through the network.

```
Prediction

↓

Dense(1)

↓

Dense(32)

↓

LSTM Layer

↓

Embedding Layer
```

Inside the LSTM, gradients flow backward through every time step.

```
Time Step 100

↓

Time Step 99

↓

Time Step 98

↓

...

↓

Time Step 2

↓

Time Step 1
```

This backward flow updates the parameters responsible for the Forget Gate, Input Gate, Candidate Cell State, and Output Gate.

---

## 8.49 What Parameters Are Updated?

Unlike a Simple RNN, an LSTM contains several groups of trainable weights.

During backpropagation, TensorFlow updates:

- Forget Gate weights
- Input Gate weights
- Candidate Cell State weights
- Output Gate weights
- Embedding Layer weights
- Dense Layer weights

Each update is computed using the gradients obtained during Backpropagation Through Time.

The Adam optimizer then adjusts these weights to reduce the prediction error.

---

## 8.50 Learning Better Memory

Initially, the gates produce values based on randomly initialized weights.

For example:

```
Forget Gate

↓

0.45
```

After many training iterations, the model learns when to retain information and when to discard it.

For important sentiment words such as:

```
excellent

amazing

terrible

worst
```

the gates gradually learn to preserve their influence in the Cell State.

For less useful information, such as dates or other details that do not help predict sentiment, the gates learn to reduce their influence.

As training progresses, the Cell State becomes a more effective long-term memory.

---

## 8.51 Why Does LSTM Learn Better Than a Simple RNN?

Both RNNs and LSTMs use Backpropagation Through Time.

The difference is that an LSTM has a dedicated Cell State that allows gradients to flow more effectively across long sequences.

Instead of repeatedly overwriting the same memory, the Cell State provides a more stable path for information.

As a result:

- Important information is preserved for longer.
- The Vanishing Gradient Problem is greatly reduced.
- Long-term dependencies are learned more effectively.

---

## 8.52 Comparison of Learning

| Simple RNN | LSTM |
|------------|------|
| One Hidden State | Hidden State + Cell State |
| Learns using BPTT | Learns using BPTT |
| Suffers from Vanishing Gradient | Greatly reduces Vanishing Gradient |
| Limited long-term memory | Better long-term memory |
| Simpler architecture | More sophisticated architecture |

---

## 8.53 Key Takeaways

- LSTM learns using Backpropagation Through Time.
- The prediction error flows backward through every time step.
- All gates contain trainable weights that are updated during training.
- The Cell State provides a more stable path for long-term information.
- LSTM reduces the Vanishing Gradient Problem and learns long-range dependencies more effectively than a Simple RNN.

## 8.54 Interview Questions

The following are common interview questions related to the LSTM architecture used in this project.

---

### Basic Questions

### Q1. What is an LSTM?

**Answer:**

LSTM (Long Short-Term Memory) is a specialized type of Recurrent Neural Network (RNN) designed to learn long-term dependencies in sequential data.

Unlike a Simple RNN, an LSTM maintains both a Hidden State and a Cell State, allowing it to preserve important information over longer sequences.

---

### Q2. Why was LSTM introduced?

**Answer:**

Simple RNNs suffer from the Vanishing Gradient Problem, making it difficult to remember information from earlier time steps.

LSTM introduces a Cell State and gating mechanisms that help preserve important information over long sequences.

---

### Q3. What are the two memories in an LSTM?

**Answer:**

LSTM maintains two memories:

- Hidden State (Short-Term Memory)
- Cell State (Long-Term Memory)

The Hidden State represents the current output, while the Cell State carries long-term information across the sequence.

---

### Q4. What is the purpose of the Cell State?

**Answer:**

The Cell State stores important information over long sequences.

It allows information to flow through the network with minimal modification, helping the model remember important context.

---

### Q5. What is the input shape to the LSTM in your project?

**Answer:**

After the Embedding Layer, the input shape is:

```
(batch_size,100,128)
```

For our project:

```
(128,100,128)
```

Meaning:

- 128 reviews
- 100 tokens per review
- 128 embedding features per token

---

### Intermediate Questions

### Q6. What are the three gates in an LSTM?

**Answer:**

The three gates are:

- Forget Gate
- Input Gate
- Output Gate

Together, these gates control what information should be forgotten, stored, and passed to the next time step.

---

### Q7. What is the purpose of the Forget Gate?

**Answer:**

The Forget Gate determines what information should be removed from the previous Cell State.

It produces values between 0 and 1 for every feature.

Values close to 1 preserve information.

Values close to 0 discard information.

---

### Q8. What is the purpose of the Input Gate?

**Answer:**

The Input Gate decides what new information should be stored in the Cell State.

It works together with the Candidate Cell State to update long-term memory.

---

### Q9. What is the purpose of the Output Gate?

**Answer:**

The Output Gate determines which information from the updated Cell State should become the Hidden State.

The Hidden State is then passed to the next time step and can also be used as the output of the current time step.

---

### Q10. Why does LSTM have both a Cell State and a Hidden State?

**Answer:**

The Cell State stores long-term information, while the Hidden State represents short-term information and the current output.

Separating these two memories allows LSTM to preserve important information for much longer than a Simple RNN.

---

### Q11. What does `return_sequences=False` do?

**Answer:**

It returns only the final Hidden State.

Output Shape:

```
(batch_size,64)
```

This is appropriate for sentiment classification because one prediction is required for each review.

---

### Q12. What does `return_sequences=True` do?

**Answer:**

It returns the Hidden State from every time step.

Output Shape:

```
(batch_size,100,64)
```

This is commonly used when another LSTM layer follows.

---

### Q13. How does batch processing work in an LSTM?

**Answer:**

The batch dimension is processed in parallel, while the time dimension is processed sequentially.

For example:

```
Input

(128,100,128)

↓

Time Step 1

↓

128 Hidden States

↓

Time Step 2

↓

128 Updated Hidden States

↓

...

↓

Time Step100
```

---

### Q14. Why did you use `return_sequences=False`?

**Answer:**

The project predicts one sentiment for the entire review.

Only the final Hidden State is needed because it summarizes the entire sequence.

---

### Advanced Questions

### Q15. How does an LSTM learn?

**Answer:**

LSTM learns using Backpropagation Through Time (BPTT).

During training, gradients are propagated backward through all time steps, updating the weights of the Forget Gate, Input Gate, Candidate Cell State, Output Gate, Embedding Layer, and Dense Layers.

---

### Q16. Does LSTM completely eliminate the Vanishing Gradient Problem?

**Answer:**

No.

LSTM greatly reduces the Vanishing Gradient Problem by introducing the Cell State and gates, but it does not eliminate it completely.

For extremely long sequences, gradients may still diminish, although much less severely than in a Simple RNN.

---

### Q17. Why is LSTM better than a Simple RNN?

**Answer:**

LSTM can remember important information over much longer sequences.

The Cell State provides a stable path for information, and the gates intelligently control memory updates.

Simple RNNs overwrite their hidden state repeatedly and therefore struggle with long-term dependencies.

---

### Q18. What is the main disadvantage of LSTM?

**Answer:**

LSTM has a more complex architecture with multiple gates.

As a result:

- More trainable parameters
- Slower training
- Higher computational cost
- Greater memory usage

---

### Q19. Explain the architecture used in your project.

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

Embedding

↓

LSTM(64)

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

---

### Q20. When would you choose LSTM over a Simple RNN?

**Answer:**

LSTM is preferred when the task requires learning long-term dependencies, such as:

- Sentiment Analysis
- Machine Translation
- Speech Recognition
- Text Generation
- Time Series Forecasting

---

## 8.55 Common Mistakes

- Confusing the Cell State with the Hidden State.
- Thinking the Cell State is the model output.
- Assuming the gates are manually programmed instead of learned during training.
- Believing LSTM completely removes the Vanishing Gradient Problem.
- Forgetting that `return_sequences=True` changes the output shape.
- Assuming every gate uses different input data.

---

## 8.56 Best Practices

- Use an Embedding Layer before the LSTM when working with text.
- Apply sequence padding so all reviews have the same length.
- Use `return_sequences=False` for sequence classification tasks.
- Add Dropout to reduce overfitting.
- Tune the number of LSTM units according to the complexity of the dataset.
- Save both the trained model and tokenizer for inference.
- Monitor validation accuracy and use Early Stopping during training.

---

## 8.57 Chapter Summary

In this chapter, we explored the complete working of Long Short-Term Memory (LSTM) networks.

We learned why LSTMs were developed to overcome the limitations of Simple RNNs, how the Cell State and Hidden State work together, and how the Forget Gate, Input Gate, Candidate Cell State, and Output Gate control the flow of information.

We also studied batch processing, `return_sequences`, Backpropagation Through Time, and how these concepts were applied in our Amazon Review Sentiment Analysis project.

Compared with a Simple RNN, an LSTM provides a more reliable mechanism for learning long-term dependencies, making it one of the most widely used architectures for sequential data.
