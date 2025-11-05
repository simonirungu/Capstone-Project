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

3. Data Understanding

The dataset used for the AfyaMind project is publicly available via HuggingFace and includes text, emotion labels and metadata such as comment ID, subreddit, author and timestamp. The dataset comprises 58,009 Reddit comments annotated with 27 emotion categories and Neutral, sourced from Reddit. The emotions range from joy, sadness and anger to surprise, fear and love. 
The dataset was stored in CSV format and read into a panda DataFrame for analysis.





