# BBC News Text Classification

## Overview

This project implements a text classification system to categorize BBC news articles into five distinct categories: Business, Entertainment, Politics, Sports, and Technology. The model leverages Natural Language Processing (NLP) techniques and machine learning algorithms to preprocess text data and make predictions.

## Table of Contents

- [Features](#features)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Usage](#usage)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Created By](#created-by)

## Features

- **Text Preprocessing**: Utilizes NLTK for tokenization, stop word removal, and lemmatization.
- **Model Training**: Implements a machine learning model to classify text.
- **Web Interface**: Built using Gradio for easy interaction with the model.
- **Real-time Predictions**: Users can input text and receive instant classification results.

## Technologies Used

- **Python**: The primary programming language.
- **NLTK**: Natural Language Toolkit for text processing.
- **scikit-learn**: Machine learning library for model implementation.
- **Gradio**: A library to create user-friendly web interfaces.
- **Pickle**: For saving and loading model objects.

## Dataset

The dataset used in this project consists of BBC news articles collected from the BBC website. Each article is labeled with one of five categories. You can find the dataset [here](https://www.bbc.co.uk/news/).

## Getting Started

### Prerequisites

Ensure you have the following installed on your system:

- Python 3.6 or higher
- pip (Python package manager)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/MuhammedTariqAbdelaal/BBC-News-Text-Classification.git
   cd BBC-News-Text-Classification

2. Install the required packages:
   ```bash
   pip install -r requirements.txt

### Usage

1. **Load the model**: Ensure that you have the `model.p` file in the root directory of the project, which contains the trained model and vectorizer.

2. **Run the application**: Execute the following command in your terminal:
   ```bash
   python app.py
   
3. **Interact with the web interface**: Open your web browser and navigate to `http://localhost:7860`. You can input a news article, and the tool will classify it into one of five categories.

## Model Training

To train your model, you can use the train_model.py script. This script includes the process of loading the dataset, preprocessing the text, training the machine learning model, and saving the model and vectorizer using Pickle.

### Steps:

1. Modify the `bbc-text-classification.ipynb` file to include the path to your dataset.
2. Run the script:
     ```bash
     python train_model.py

## Evaluation

Evaluate the model's performance using metrics such as accuracy, precision, recall, and F1 score. The evaluation results will provide insights into how well the model is classifying news articles.

## Results

The model achieved an accuracy of approximately 97% on the validation dataset. Further evaluation metrics can be found in the evaluation section of the code.

## Contributing

Contributions are welcome! If you would like to contribute to this project, please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

## Acknowledgments

- NLTK: For providing powerful tools for text processing.
- Gradio: For enabling easy web interface creation.
- The authors of the datasets used for this project.

## Created By: 

Muhammed Tariq Abdelaal Aboseif
