# Toxic Comment Classification Service

## 1. Description of the Problem

Online discussion platforms and social media are often plagued by toxic behavior, ranging from rude comments to hate speech and threats. Manual moderation is unscalable, and simple keyword filtering is often ineffective against nuanced language.

This project builds an **end-to-end Machine Learning service** to automatically detect various types of toxicity in user-generated comments.

The goal is to classify text into one or more of the following **six categories** (Multi-label classification):
1.  `toxic`
2.  `severe_toxic`
3.  `obscene`
4.  `threat`
5.  `insult`
6.  `identity_hate`

The project utilizes Natural Language Processing (NLP) techniques, specifically **TF-IDF vectorization** combined with a **Linear Support Vector Machine (SVM)** (via SGD with Hinge Loss), wrapped in a **One-Vs-Rest strategy**, which is an implementation of the **Binary Relevance** method of solving multi-label classification problems. The final model is deployed as a web service using **FastAPI** and **Docker**.

---

## 2. Project Structure

* **`data/`**: Folder containing the dataset (see Data section).
* **`notebook.ipynb`**: Jupyter notebook covering Data Wrangling, EDA (N-gram analysis, length distribution), Preprocessing pipeline development, Model Selection, Hyperparameter tuning, and Threshold optimization.
* **`train.py`**: Script to train the final pipeline on the full dataset and save the model artifact (`.pkl`).
* **`predict.py`**: The deployment script using FastAPI. It loads the model and serves predictions via a REST API.
* **`cleaning.py`**: A helper module containing the custom text preprocessing logic (cleaning regex, Spacy lemmatization, etc.) used by both training and inference.
* **`contractions.py`**: A helper dictionary for expanding English contractions.
* **`test.py`**: A simple script to send a request to the running service and verify predictions.
* **`Dockerfile`**: Configuration for containerizing the application.
* **`pyproject.toml` / `uv.lock`**: Dependency management files (using `uv`).
* **`toxic_comment_prediction_model.pkl`**: The saved Scikit-Learn pipeline.
* **`thresholds.json`**: Optimized probability thresholds for each class derived from the notebook analysis.

---

## 3. Data

The dataset used is from the **Kaggle Jigsaw Toxic Comment Classification Challenge**.

### Instructions to download:
1.  Go to the [Kaggle Competition Page](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data).
2.  Download `jigsaw-toxic-comment-classification-challenge.zip`.
3.  Create a folder named `data` in the root directory of this project.
4.  Unzip `train.csv.zip` into the `data/` folder.
    > **Note:** The training script expects `data/clean_toxic_comment_dataset.csv`. If running from scratch, the `notebook.ipynb` generates this cleaned file.

---

## 4. Dependency Management

This project uses **`uv`** (a fast Python package installer and resolver) for dependency management.

### Prerequisites
* Python 3.12
* `uv` installed (`pip install uv`)

### Installing Dependencies
To install the environment locally:
```bash
uv sync
```

Alternatively, if you are using standard pip:

**Install dependencies:**

```bash
pip install pandas numpy scikit-learn spacy nltk fastapi uvicorn joblib
```

#### Installing the spaCy Model

The spaCy English model is **not included in the repository**. You can obtain it in either of the following ways:

##### Option A — Install from the internet (recommended)

**Using `uv`:**

```bash
uv pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
```

##### Option B — Use a local wheel file (if you already have it)

Place `en_core_web_sm-3.8.0-py3-none-any.whl` in the project directory and install with uv:

```bash
uv pip install en_core_web_sm-3.8.0-py3-none-any.whl
```

Or with standard pip:
```bash
pip install en_core_web_sm-3.8.0-py3-none-any.whl
```

---

## 5. Instructions to Run the Project

### A. Running Locally (Development)

#### Preprocessing & Training  
If you want to reproduce the model training process:

```bash
# Ensure data is in data/train.csv
python train.py
```

This will generate **`toxic_comment_prediction_model.pkl`**.

#### Running the Server  
Start the FastAPI service:

```bash
uvicorn predict:app --reload --host 0.0.0.0 --port 8080
```

#### Testing the Service  
In another terminal:

```bash
python test.py
```

---

### B. Running with Docker (Production)

The project includes a **Dockerfile** that installs system dependencies (NLTK data, SpaCy models) and serves the app.

**Build the image:**

```bash
docker build -t toxic-comment-prediction .
```

**Run the container:**

```bash
docker run -it --rm -p 8080:8080 toxic-comment-prediction
```

**Interact with the service:**

- Use `python test.py`, or  
- Open Swagger UI:  
  `http://localhost:8080/docs`

---

## 6. Model and Approach

### Text Preprocessing (`cleaning.py`)

The cleaning pipeline performs:

- HTML tag removal  
- IP address and URL removal  
- Contraction expansion (e.g., `"don't"` → `"do not"`)  
- Lemmatization with SpaCy  
- Stopword removal (while preserving *not*, *no*)  

### Model Selection

The `notebook.ipynb` compares Logistic Regression vs Linear SVM within two methods for solving multilabel classification problems: **`Binary Relevance`**, implemented using scikit-learn's **`OneVsRestClassifier`** and **`Classifier Chains`** implemented by scikit-learn's **`ClassifierChain`**. Four models were trained.

**Final Model:**

- `OneVsRestClassifier` with `SGDClassifier` (hinge loss, SVM)
- `CalibratedClassifierCV` for probability outputs  
- `TfidfVectorizer` with N-grams (1,2)  
- Custom per-label **thresholds** (stored in `thresholds.json`) to maximize F1 due to class imbalance  

---

## 7. Deployment Interaction

### API Endpoint  
- **URL:** `/predict`  
- **Method:** `POST`

### Sample Request Body

```json
{
  "text": "You're funny. Ugly? We're dudes on computers, moron. You're quite astonishingly stupid."
}
```

### Sample Response

```json
{
  "toxicity_probability": {
    "toxic": {
      "score": 0.9999493157158548,
      "is_toxic": true
    },
    "severe_toxic": {
      "score": 0.02834308396669576,
      "is_toxic": false
    },
    "obscene": {
      "score": 0.9639879904078187,
      "is_toxic": true
    },
    "threat": {
      "score": 0.0017378427213896295,
      "is_toxic": false
    },
    "insult": {
      "score": 0.9967625674780264,
      "is_toxic": true
    },
    "identity_hate": {
      "score": 0.02195613626856109,
      "is_toxic": false
    }
  }
}
```

**NOTE**: *(`is_toxic` is determined using thresholds from `thresholds.json` during inference.)*