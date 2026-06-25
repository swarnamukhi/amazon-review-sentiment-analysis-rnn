📖 Chapter 1 - Project Introduction


## 1.1 Project Overview

Sentiment Analysis is one of the most widely used applications of Natural Language Processing (NLP). It involves analyzing textual data to determine the emotional tone or opinion expressed by a user.

In this project, an end-to-end Deep Learning-based sentiment analysis system is developed to classify Amazon customer reviews as either **Positive** or **Negative**.

The project begins with preprocessing raw customer reviews by cleaning the text, tokenizing words into numerical representations, and padding sequences to create fixed-length inputs. An Embedding layer is then used to learn meaningful word representations, followed by Recurrent Neural Networks (RNN) and Long Short-Term Memory (LSTM) models to capture sequential dependencies in the text.

To achieve the best performance, Bayesian Hyperparameter Optimization is applied to automatically determine the optimal model configuration. The final model is trained using the best hyperparameters, evaluated on unseen test data, and saved along with the tokenizer for future predictions.
