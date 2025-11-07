# Capstone-Project: AfyaMind Emotion Detection Project 
Team: Simon Irungu, John Nyangoya, Eugene Mallah, Sophie Muchiri, Bob-lewis 

# Project Overview

The AfyaMind project develops an automated emotion detection model to classify text-based data into distinct emotion categories. This system supports mentah health professionals, researchers, and digital wellness organizations by identifying emotional states in written communication. It enables early interventions, empathetic responses, and large-scale emotional trend analysis.

Here is the link to our <a href="https://www.canva.com/design/DAG3t8V_d5M/O0fOdKRHZbzChupC_rGeoA/view?utm_content=DAG3t8V_d5M&utm_campaign=designshare&utm_medium=link2&utm_source=uniquelinks&utlId=he656109d9f" target="_blank">presentation slides</a>

Here is the link to our <a href="https://afyamindapp.streamlit.app/" target="_blank"> deployed app.</a>

# Business Understanding
In the digital era, understanding emotions expressed in text is essential for enhancing user interactions, improving customer support, and analyzing public sentiment. This project details the end-to-end process: from essential text preprocessing and exploratory data analysis (EDA) to the implementation and optimization of a deep learning model. Crucially, the focus here is on multi-label classification, moving beyond binary sentiment to capture the subtle complexity of human communication, where emotions like 'confusion' and 'excitement' can coexist. The resulting system, deployed via a Streamlit application, transforms raw text into actionable emotional profiles, enabling personalized and context-aware responses, as well as application to a mental health chatbot known as `AfyaMind.`

# Objectives:

*Main Objective:* To build and evaluate a machine learning model that can automatically classify text into emotional categories using the GoEmotions dataset.

*Specific Objectives*

To explore and understand the distribution of emotions in the GoEmotions dataset.

To transform textual data into numerical form using techniques such as TF-IDF, or contextual embeddings (e.g., BERT).

To build and compare baseline models (e.g., Logistic Regression, Random Forest, Naive Bayes) with deep learning models (e.g., BERT).

To evaluate models using metrics appropriate for multi-label classification (e.g., F1-score, Precision, Recall)

To deploy the model on Streamlit and intergrate it with an existing model to provide intervention.

*Success Criteria:* Evaluated via performance metrics like macro F1-score and recall. Target: Macro F1-score >= 0.75 for balanced accuracy across emotion classes.


# Data Understanding

The dataset is the publicly available GoEmotions from Hugging Face, consisting of 58,009 Reddit comments annotated with 27 emotion categories plus "Neutral."Emotions include joy, sadness, anger, suprise, fear, love, and more.

Dataset Structure:

- Format: CSV, loaded into Pandas DataFrame.
- Columns: 'text', 'emotion', metadata(e.g., comment ID, subreddit, author, timestamp).
- Rows: Each represents a labeled text instance.
- Label Distribution: Common emotions like 'joy' and 'sadness' dominate; rarer ones like 'fear' and 'suprise' are underrepresented.

Data Quality Checks:

- No null/empty values in key columns.
- Removed Duplicates
- Verified labels against predifined categories
- Sampled texts for labeling accuracy

## Important Visualizations

<img width="813" height="479" alt="image" src="https://github.com/user-attachments/assets/facba704-5a42-4064-b5bd-bfd383cbe0da" />

<img width="819" height="448" alt="image" src="https://github.com/user-attachments/assets/94aedfb7-45b2-474a-950c-70dbad6388aa" />

<img width="819" height="501" alt="image" src="https://github.com/user-attachments/assets/fffb2aba-d16c-48b4-a940-8bc09c939ef8" />


# Data Preparation

This phase transformed raw text into model-ready formats for consistency and performance.

Key Steps:

- Text Cleaning: Lowercased text; removed special characters, digits, URLs, punctuation via regex; expanded constractions; trimmed whitespace.

- Tokenization: Used NLTK word tokenizer for classical models; Hugging Face tokenizer for transformers.

- Stopword Removal: Eliminated common words (e.g., 'and', 'the', 'is') to reduce noise.

- Lemmatization: Reduced words to base forms using WordNetLemmatizer.

- Feature Extraction: TF-IDF vectorization for classical models and token embeddings (IDs, attention masks) for transformers like RoBERTa.

- Handling Missing Data: Dropped entries with missing labels; imputed auxialiary columns with median/mode.

- Train-Test Split: 80/20 stratified split to preserve class proportions.

# Modelling

## Baseline: Classical Machine Learning Models

These provided benchmarks using feature engineering (e.g., TF-IDF)

Logistic Regression: Wrapped in MultiOutputClassifier for multi-label.

Linear Support Vector Classifier (SVC): Optimal hyperplane with class weights; best baseline (Micro F1: 0.5532)

Support Vector Machine (SVM): Linear Kernel, adjusted weights for minorities

Multinomial Naive Bayes: Probabilistic, independent features; fast but minimal complexity.
```
--- Final Weighted Model Comparison ---
                 F1 Score (Weighted)  Precision (Weighted)  Recall (Weighted)
SVC Weighted                  0.5530                0.6801             0.4894
LogReg Weighted               0.5212                0.7435             0.4352

Best performing model: SVC Weighted
Best F1 score: 0.5530
```
The Linear SVC with weighted classes performed the best. This provided a baseline that our Deep Learning model had to surpass 

## Deep Learning Model

RoBERTA (Robustly Optimized BERT Pretraining Approach). RoBERTA's ability to understand the deep, bidirectional context of language allows it to capture subtle emotional nuances that simpler models miss, leading to a substantial gain in overall effectiveness.

```
Final Model Performance (7 epochs):
Training Loss: 0.0988
Validation Loss: 0.2983
Accuracy: 0.5916
F1 Micro: 0.6649
F1 Macro: 0.5930
```
The RoBERTA model is superior in capturing nuance emotions(e.g., 'sadness' vs 'fear').

# Model Implementation and Evaluation

<img width="411" height="317" alt="image" src="https://github.com/user-attachments/assets/4e4939ac-68c8-46d8-af2f-64ce957ef21d" />


<img width="539" height="415" alt="image" src="https://github.com/user-attachments/assets/610e9ec9-33a1-4e3f-9e93-1ba2bc4bb492" />

```
Testing single predictions:
'I'm absolutely thrilled with this amazing result!...' → [np.str_('joy_mapped')]
'This is disgusting and makes me so angry...' → [np.str_('disgust_mapped')]
'I feel scared and nervous about what might happen ...' → [np.str_('fear_mapped')]
'What a pleasant surprise! I didn't expect this...' → [np.str_('surprise_mapped')]
'I'm feeling pretty neutral about this situation...' → [np.str_('neutral_mapped')]
```
The model was able to predict unseen data into the correct emotions.

# Recommendations

Prioritize Minority Classes: Boost low Macro F1 (0.4431) with advanced data augmentation.

Optimize for Production: Use lighter variants (DistilBERT/RoBERTa-base) and quantization (e.g., ONNX) for low-latency Streamlit deployment.

Risk Mitigation: Add binary "AfyaMind Alert" classifier for high-risk (sadness + fear) texts.

Emotion-to-Action Framework: Define responses per emotion (e.g., High Joy: "Share a success story").

Monitoring: Track data/model drift; alert on 5% Micro F1 drop.

Document Ambiguities: Analyze top co-occuring emotion pairs for UX insights.

# Conclusion

Model Selection: The RoBERTA deep learning model ultimately achieved the best balance of performance, with a Micro F1 score of 0.579, slightly outperforming the best classical machine learning model, Linear SVC (F1 score of 0.5532).

Overall Performance: The moderate F1 scores across all models highlight the inherent complexity of multi-label emotion classification, which is challenging due to overlapping emotions, varying text complexity, and significant class imbalance in the dataset.

Class Imbalance Handling: Applying class weights resulted in only marginal improvement for the Logistic Regression model and no significant change for Linear SVC, suggesting that the chosen models inherently manage moderate imbalance reasonably well or that more advanced deep learning techniques are required to significantly address the minority classes.



    























