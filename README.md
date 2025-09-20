# English to French Translation Model

This project implements an end-to-end deep learning pipeline for translating English sentences to French using a Sequence-to-Sequence (Seq2Seq) model with an LSTM-based Encoder-Decoder architecture.

## Dataset

The model is trained on a dataset of English-French sentence pairs.
* **Description:** The dataset contains numerous pairs of sentences, one in English and its corresponding translation in French.
* **Source:** The dataset is expected to be a CSV file named `eng_-french.csv`. You can download it from here or the following link: https://www.kaggle.com/datasets/devicharith/language-translation-englishfrench Please place this file in a folder named `datasets` in the root of the project directory.

## Model Architecture

The core of this project is a Seq2Seq model built with TensorFlow and Keras.

* **Encoder:** The encoder processes the input English sentences. It consists of an `Input` layer, an `Embedding` layer to convert words into dense vectors, a `Dropout` layer for regularization, and an `LSTM` layer that captures the sentence context and generates a final state vector.
* **Decoder:** The decoder uses the final state of the encoder to generate the French translation. It includes its own `Input` and `Embedding` layers, a `Dropout` layer, and an `LSTM` layer. The decoder's LSTM is initialized with the encoder's state vectors, allowing it to start decoding with the context of the English sentence.
* **Output Layer:** A final `Dense` layer with a softmax activation produces the probability distribution for each word in the French vocabulary, predicting the next word in the sequence.

## Features

* **End-to-End ASR Pipeline:** From data loading to preprocessing and model training, the notebook provides a complete pipeline.
* **Text Preprocessing:** Includes custom functions for cleaning text by converting to lowercase and removing special characters.
* **Tokenization and Padding:** Utilizes Keras's `Tokenizer` and `pad_sequences` for efficient data preparation for the model.
* **Callback Functions:** Incorporates `EarlyStopping`, `ModelCheckpoint`, and `ReduceLROnPlateau` to prevent overfitting, save the best model, and dynamically adjust the learning rate during training.
* **Inference Function:** A dedicated function is provided to take a new English sentence and translate it using the trained model.

## Usage

To run this project, follow these steps:

1.  **Download the Dataset:** Download the `eng_-french.csv` file from the provided link and place it inside a new folder named `datasets`.
2.  **Install Dependencies:** Ensure you have all the necessary libraries installed. You can install them using pip:
    ```bash
    pip install pandas tensorflow scikit-learn matplotlib
    ```
3.  **Run the Notebook:** Open the `English-French-NMT-Translation-Model.ipynb` file in a Jupyter environment (e.g., Jupyter Notebook, JupyterLab, or Google Colab).
4.  **Execute Cells:** Run all the cells in the notebook sequentially. The model will automatically train, and the final cells will demonstrate its usage by translating a sample sentence.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
