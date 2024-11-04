# BBC-News-Text-Classification
A Text Classification model to categorize BBC news articles into predefined topics using natural language processing and machine learning techniques.


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
