# Capstone-Project: AfyaMind Emotion Detection Project 
Team: Simon Irungu, John Nyangoya, Eugene Mallah, Sophie Muchiri, Bob-lewis 

#Project Overview

The AfyaMind project develops an automated emotion detection model to classify text-based data into distinct emotion categories. This system supports mentah health professionals, researchers, and digital wellness organizations by identifying emotional states in written communication. It enables early interventions, empathetic responses, and large-scale emotional trend analysis.


#Key Objectives:

Build and evaluate a machine learning model for automatic text classification into emotional categories using the GoEmotions dataset.

Explore and understand emotion distribution in the dataset.

Compare baseline models with deep learning approaches.

Evaluate models using multi-label classification metrics.

Deploy the model on Streamlit and intergrate with intervention systems.

Generate actionable insights for digital health and AI-driven emotional intelligence.

#Success Criteria: Evaluated via performance metrics like macro F1-score and recall. Target: Macro F1-score >= 0.75 for balanced accuracy across emotion classes.


# Business Understanding

The project aims to create a robust emotion detection system for real-world applications in mental health. By analyzing text from sources like Reddit comments, it identifies emotions to inform strategies in digital wellness.

# Data Understanding

The dataset is the publicly available GoEmotions from Hugging Face, consisting of 58,009 Reddit comments annotated with 27 emotion categories plus "Neutral."Emotions include joy, sadness, anger, suprise, fear, love, and more.

Dataset Structure:

Format: CSV, loaded into Pandas DataFrame.
Columns: 'text', 'emotion', metadata(e.g., comment ID, subreddit, author, timestamp).
Rows: Each represents a labeled text instance.
Label Distribution: Common emotions like 'joy' and 'sadness' dominate; rarer ones like 'fear' and 'suprise' are underrepresented.
Volume: Thousands of entries, suitable for ML and DL models.

Exploratory Data Analysis (EDA) Insights:

#Variations in spelling, abbreviations, slang.
#Emojis, symbols, and varying text lengths (single words to multi-sentence)
#Class Imbalance across emotions.


Data Quality Checks:

#No null/empty values in key columns.
#Removed Duplicates
#Varified labels against predifined categories
#Sampled texts for labeling accuracy


# Data Preparation

This phase transformed raw text into model-ready formats for consistency and performance.

Key Steps:

Text Cleaning: Lowercased text; removed special characters, digits, URLs, punctuation via regex; expanded constractions; trimmed whitespace.

Tokenization: Used NLTK word tokenizer for classical models; Hugging Face tokenizer for transformers.

Stopword Removal: Eliminated common words (e.g., 'and', 'the', 'is') to reduce noise.

Lemmatization: Reduced words to base forms using WordNetLemmatizer.

Label Encoding: Converted emotions to intergers via scikit-learn's LabelEncoder.

Feature Extraction:

    TF-IDF vectorization for classical models

    Token embeddings (IDs, attention masks) for transformers like RoBERTa.

Handling Missing Data: Dropped entries with missing labels; imputed auxialiary columns with median/mode.

Class Imbalance Mitigation: SMOTE for classical models; weighted loss for transformers.

Train-Test Split: 80/20 stratified split to preserve class proportions.

Verification: Checked feature matrix dimensions post-processing.


#Modeling Approaches

Baseline: Classical Machine Learning Models

These provided benchmarks using feature engineering (e.g., TF-IDF)

#Logistic Regression:   Tuned with GridSearchCV (C, solver, iterations); wrapped in MultiOutputClassifier for multi-label.

#Linear Support Vector Classifier (SVC): Optimal hyperplane with class weights; best baseline (Micro F1: 0.5532)

#Support Vector Machine (SVM): Linear Kernel, adjusted weights for minorities

#Naive Bayes: Probabilistic, independent features; fast but minimal complexity.

Final Solution: Deep Learning Model

#BERT (Bidirectional Encoder Representations from Transformers): Fine-tuned on 7-label dataset; excels in contextual understanding.

        Achieved highest performance (Micro F1: 0.5930, Hamming Loss: 0.0834)

        Selected for deployment due to nuance capture.



# Model Implementation and Evaluation

Evaluated using accuracy, precision, recall, F1-score (macro-averaged for imbalance), and confusion matrices.

Baseline Performance

Logistic Regression: Micro F1: 0.5487, Macro F1: 0.5551, Hamming Loss: 0.1097.

Linear SVC: Micro F1: 0.5532, Hamming Loss: 0.01065 (stongest baseline)

BERT Model Performance

Micro F1: 0.5930 (highest accuracy)

Hamming Loss: 0.0834 (lowest error rate < 8.5%).

Superior in capturing nuances; reduced ambiguities (e.g., 'sadness' vs 'fear').

#Diagnostics:

Common misclassifications: Overlaps in linguistic expressions; short/ambiguous texts as 'neutral'.

Hyperparameter Tuning: Early stopping after ~5 epochs to prevent overfitting.

Met success threshold; ready for deployment.

# Recommendations

Prioritize Minority Classes: Boost low Macro F1 (0.4431) with advanced data augmentation.

Optimize for Production: Use lighter variants (DistilBERT/RoBERTa-base) and quantization (e.g., ONNX) for low-latency Streamlit deployment.

Risk Mitigation: Add binary "AfyaMind Alert" classifier for high-risk (sadness + fear) texts.

Emotion-to-Action Framework: Define responses per emotion (e.g., High Joy: "Share a success story").

Monitoring: Track data/model drift; alert on 5% Micro F1 drop.

Document Ambiguities: Analyze top co-occuring emotion pairs for UX insights.

# Conclusion

The BERT model forms a stable foundation with real performance (Micro F1: 0.5930, Hamming Loss: 0.0834). AfyaMind is an intelligent asset for digital mental health, ready for frontline deployment. Recommend approving Phase 1 actions for reliability, cost reduction, and mission fulfillment.


    























