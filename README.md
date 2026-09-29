# 🛡️ AI Content Moderator
![AI Moderator – Safe Content](screenshots/app-safe.png)

![AI Moderator – Harmful Content](screenshots/app-harmful.png)

![AI Moderator – Burmese Content](screenshots/app-burmese.png)
An AI-powered content moderation application built with Python, Hugging Face Transformers, and Streamlit.

The application analyzes user-generated comments in **English and Burmese** and provides a risk assessment to help identify potentially harmful content.

## 🚀 Features

- English and Burmese text moderation
- AI-powered toxicity detection
- Risk score calculation
- Low, Medium, and High risk levels
- Allow, Review, and Remove moderation decisions
- Category-level toxicity scores
- Interactive Streamlit interface
- Burmese moderation dataset preparation and experimentation
- Model evaluation using accuracy, precision, recall, and F1 score

## 🧠 Model

The main application uses the Hugging Face:

`unitary/toxic-bert`

The project also includes an experimental Burmese moderation pipeline using XLM-RoBERTa and a curated Burmese moderation dataset.

The Burmese model was developed and evaluated separately as part of the project's multilingual moderation experimentation.

## 🏗️ Project Structure

```text
ai-moderator/
│
├── app.py
├── README.md
├── .gitignore
│
└── src/
    ├── model.py
    ├── predictor.py
    ├── evaluate_burmese_model.py
    ├── prepare_burmese_dataset.py
    ├── prepare_burmese_dataset_v2.py
    ├── split_burmese_dataset.py
    ├── split_burmese_dataset_v2.py
    ├── train_burmese_model.py
    ├── train_burmese_model_v2.py
    ├── test_burmese_model_v2.py
    ├── test_burmese_predictions.py
    ├── test_burmese_predictions_v2.py
    └── analyze_burmese_model_v2.py
