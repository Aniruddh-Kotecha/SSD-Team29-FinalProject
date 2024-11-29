# Python Code Tokenization and Next Word Prediction

## Overview

This project focuses on building a deep learning-based next-word prediction system for Python code using an LSTM (Long Short-Term Memory) neural network and GRU (Gated Recurrent Unit). The pipeline involves tokenizing Python scripts, converting tokens to numerical IDs, and training an LSTM-based recurrent neural network to predict the next token in a given sequence.

## Project Workflow

### 1. Tokenization

The project uses the Python tokenize library to parse Python code into tokens:

- `tokenize_file`: Tokenizes a single Python file, removing comments and empty lines.
- `tokenize_directory`: Tokenizes all Python files in a specified directory.

### 2. Vocabulary Creation

A vocabulary is built from all unique tokens found in the tokenized dataset. Each token is assigned a unique integer ID for use as input to the model.

### 3. Sequence Preparation

Input-output pairs are created:

- Input: A sequence of token IDs of fixed length
- Output: The next token ID

**Example:**
Input: `[for, i, in, range]` → Output: `:`

### 4. LSTM Model

The model architecture includes:

- Embedding layer: Converts token IDs to dense vector representations
- Two LSTM layers: Capture sequential patterns in token sequences
- Dropout layer: Prevents overfitting
- Dense layer: Outputs the next token probability distribution using a softmax activation

## Functions

- `tokenize_file(file_path)`: Tokenizes a single Python file, filtering out unnecessary tokens (e.g., comments, empty lines)
- `tokenize_directory(directory_path)`: Processes all Python files in a specified directory and returns their tokens
- `build_vocabulary(tokenized_data)`: Builds a dictionary mapping tokens to unique IDs
- `convert_tokens_to_ids(tokenized_data, vocab)`: Converts tokens to corresponding numerical IDs using the vocabulary
- `prepare_sequences(token_ids_data, sequence_length=4)`: Prepares training sequences of fixed length

## Model Architecture Details

### Embedding Layer

- Input: Total vocabulary size and embedding dimensions (100 in this case)

### LSTM Layers

- First layer (`return_sequences=True`): Captures patterns in sequences
- Second layer: Condenses the sequence into a single representation

### Dropout Layer

- Adds regularization by randomly deactivating neurons during training

### Dense Layer

- Outputs probabilities for all words in the vocabulary using the softmax activation function

## Dependencies

The following Python libraries are required:

- `os` (for directory traversal)
- `tokenize` (for tokenization of Python code)
- `numpy` (for numerical operations)
- `sklearn` (for splitting datasets)
- `tensorflow` (for deep learning model creation and training)
- `pickle` (for saving the tokenizer)

### Installation

```bash
pip install tensorflow numpy scikit-learn
```

## Usage

### 1. Tokenize Python Files

```python
directory_path = "Folder_name"
tokenized_data = tokenize_directory(directory_path)
```

### 2. Build Vocabulary

```python
vocab = build_vocabulary(tokenized_data)
```

### 3. Prepare Training Data

```python
x, y = prepare_sequences(token_ids_data, sequence_length=4)
```

### 4. Train the Model

```python
history = model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test), verbose=1)
```

### 5. Save the Model and Tokenizer

```python
model.save("Model_name.h5")
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)
```

### 6. Predict the Next Word

```python
input_text = "for i in"
next_word = predict_next_word_custom(model, input_text)
print(f"Next Word Prediction: {next_word}")
```

## Project Highlights

- **Dynamic Tokenization**: Handles Python code files of varying formats
- **Self-Supervised Learning**: Learns word relationships from the code context itself
- **Scalability**: Can be adapted for larger datasets and more complex architectures

## Future Improvements

- Use pre-trained embeddings like CodeBERT or CodeT5 for enhanced performance
- Extend the model to handle other programming languages
- Optimize the architecture using hyperparameter tuning

## Example Output

For the input sequence "for i in", the model might predict:

```
Next Word Prediction: range
```

## Project Contributors

### 1. Tanmai Shah

### 2. Priyanshu Jha

### 3. Ramachandran Moorthy

### 4. Aniruddh Kotecha
