## Fake News Detection 

## Project Overview:

Fake news detection is a critical task in the modern era, where the spread of misinformation can have significant negative impacts on society. This project aims to develop a machine learning model to classify news articles as either fake or real. Below is a comprehensive guide to setting up a fake news detection project.The fake news detection project involves various steps from data collection, preprocessing, and exploratory data analysis to model training, evaluation, and deployment. The ultimate goal is to create a robust and reliable system that can accurately classify news articles as fake or real, helping to mitigate the spread of misinformation.


##Steps:

1.Data Collection:

**Datasets:You can use publicly available datasets like:
-Fake and real news dataset on Kaggle
-LIAR dataset

2.Data Preprocessing:

-Lowercasing
-Removing punctuation
-Removing stopwords
-Lemmatization

3.Exploratory Data Analysis (EDA):

**Visualization:
-Distribution of fake vs. real news.
-Common words in fake and real news.

4.Feature Engineering:

**TF-IDF Vectorization:
-Convert text data into numerical data using TF-IDF.

5. Model Training:

**Train-Test Split:
-Split the dataset into training and testing sets.

**Model Selection and Training:
-Train a classification model (e.g., Logistic Regression, Naive Bayes, SVM).

6.Model Evaluation:

**Evaluate Model Performance:
-Use metrics like accuracy, precision, recall, and F1 score.


#Machine learning Models used:

1. **Logistic Regression**
2. **Decision Tree Classifier**
3. **Gradient Boost Classifier**
4. **Random Forest Classifier**


## Dataset

We have used a labelled dataset containing news articles along with their corresponding labels (true or false). The dataset is divided into two classes:
- True: Genuine news articles
- False: Fake or fabricated news articles

## System Requirements 

Hardware :
1. 4GB RAM
2. i3 Processor
3. 500MB free space

Software :
1. Anaconda
2. Python

## Dependencies

**packages and libraries to be installed:
- Python 3
- Scikit-learn
- Pandas
- Numpy
- Seaborn
- Matplotlib
- Regular Expression

**install these dependencies using pip:

```bash
pip install pandas
pip install numpy
pip install matplotlib
pip install sklearn
pip install seaborn 
pip install re 
```

