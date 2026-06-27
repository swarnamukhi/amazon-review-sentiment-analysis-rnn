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
