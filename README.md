# Mobile Review Sentiment Analyzer

## Project Description
This project implements a machine learning pipeline designed to classify the sentiment of mobile phone reviews as either Positive or Negative. Utilizing a dataset of consumer reviews, the system integrates natural language processing (NLP) with metadata analysis to predict sentiment based on review text, helpfulness scores, and purchase verification status.

The model employs a Logistic Regression classifier within a Scikit-Learn pipeline. It features a hybrid architecture that processes unstructured text using Term Frequency-Inverse Document Frequency (TF-IDF) vectorization while simultaneously scaling numerical features for optimized classification performance.



## Technical Specifications

### Data Preprocessing
- **Text Cleaning:** Conversion to lowercase, removal of non-alphabetic characters, and filtering of English stop words.
- **Lemmatization:** Reduction of words to their base or dictionary form using the WordNet Lemmatizer.
- **Feature Engineering:** Inclusion of auxiliary features such as helpful_votes, review_length, and verified_purchase status to supplement textual data.

### Model Architecture
- **Vectorization:** TF-IDF Vectorizer with a maximum of 7,000 features, utilizing unigrams and bigrams.
- **Scaling:** Standard scaling applied to numerical metrics to ensure zero mean and unit variance.
- **Classifier:** Logistic Regression with balanced class weights to account for potential distribution inequalities between positive and negative samples.



## Implementation Requirements

### Dependencies
The project requires Python 3.8+ and the following libraries:
- pandas
- scikit-learn
- nltk
- matplotlib
- seaborn

### Installation
1. Clone the repository to your local environment.
2. Install the required packages:
   ```bash
   pip install pandas scikit-learn nltk matplotlib seaborn
3. Download the necessary NLTK corpora:
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('wordnet')

### Performance Evaluation
The model evaluation includes:
- Accuracy Score: Overall percentage of correct predictions.
- Classification Report: Detailed precision, recall, and F1-score for both sentiment classes.
- Confusion Matrix: Visual representation of model performance across the test set.

### Usage
To execute the model, ensure the mobile-reviews.csv file is in the project directory and run the main script. The script will output the classification metrics and generate visualizations for sentiment distribution and prediction errors.      
