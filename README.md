# Capstone-Project: AfyaMind Emotion Detection Project 
Team: Simon Irungu, John Nyangoya, Eugene Malla, Sophie Muchiri, Bob-lewis 

# Overview

AfyaMind is a Natural Language Processing (NLP) project that focuses on detecting emotions expressed in text.
In today’s digital environment, understanding the emotions embedded in written communication is essential for supporting mental health initiatives, social research, and digital wellness platforms.
By classifying user text into emotion categories such as joy, sadness, anger, fear, and surprise, AfyaMind aims to improve emotional awareness and provide data-driven insights to help organizations respond empathetically and effectively.

# 1. Business Understanding 

Business Problem: Mental health and digital communication platforms lack automated tools that can accurately recognize emotions from text-based messages. This limits their ability to detect negative emotional patterns and provide timely interventions.

Business Objective: To build and evaluate a machine learning model that can automatically classify text into emotional categories using the GoEmotions dataset.

Stakeholders: Mental health professionals and wellness app developers who will use emotion insights for personalized care and engagement. Research institutions analyzing emotional trends and behaviors from text data.

Success Criteria: The main metric is the Macro F1-score, targeting a minimum of 0.75 or higher to ensure balanced performance across all emotion categories regardless of data imbalance.

# 2. Data Understanding

The dataset used for the AfyaMind project is publicly available via HuggingFace and includes text, emotion labels and metadata such as comment ID, subreddit, author and timestamp. 

The dataset comprises 58,009 Reddit comments annotated with 27 emotion categories and Neutral, sourced from Reddit. The emotions range from joy, sadness and anger to surprise, fear and love. 

# 3. Data Cleaning 

In this step, basic data cleaning was done to ensure the dataset is consistent and ready for modeling. The process involved:

Checking for and handling null values

Identifying and removing duplicate rows

Standardizing column names for readability.

Cleaning texts by removing punctuation, special characters, numbers and converting all text to lowercase. 

Expanding contractions.

Tokenizing text into words using NTLK. 

Removing stopwords that do not contribute to meaning. 

Lemmatizing words using WorkNet to convert them to their base forms. 

encoding emotion labels into numeric form for model training. 

Handling class imbalance with SMOTE (Synthetic Minority Oversampling Technique) and weighted class adjustments during model training.














