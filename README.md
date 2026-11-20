# AI Fitness Trainer 🏋️‍♂️🤖

An AI-powered fitness analysis system that uses **Computer Vision, Pose Estimation, and Machine Learning** to evaluate exercise performance and classify workout form as **Good** or **Bad**.

The project analyzes human body movements from workout videos using **MediaPipe**, extracts pose landmarks, and applies machine learning techniques to identify potential exercise-form mistakes.

## 🚀 Project Overview

The AI Fitness Trainer aims to help users improve their workout technique by automatically analyzing exercise videos and evaluating movement quality.

The current implementation focuses on two exercises:

- **Bicep Curl**
- **Squat**

Videos are captured from multiple camera viewpoints to investigate how viewing angle affects exercise-form classification.

The system extracts body pose information, converts it into numerical features, and uses machine learning models to predict whether an exercise is performed correctly.

## 🎯 Project Objectives

- Detect and analyze human body movements from video.
- Extract body pose landmarks using MediaPipe.
- Convert pose landmarks into structured numerical features.
- Classify exercise performance as Good or Bad.
- Compare machine learning approaches for exercise-form assessment.
- Investigate the impact of camera viewpoints on prediction accuracy.
- Explore temporal deep-learning architectures for improved movement analysis.

## 🛠️ Technologies & Tools

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| OpenCV | Video processing and computer vision |
| MediaPipe | Human pose estimation and landmark extraction |
| NumPy | Numerical operations and feature processing |
| Pandas | Dataset management and analysis |
| Scikit-learn | Machine learning and evaluation |
| Random Forest | Baseline classification model |
| PyTorch | Deep-learning model development |
| Matplotlib / Seaborn | Visualization and performance analysis |

## 📂 Dataset

The project uses a multi-view fitness exercise video dataset containing recordings of participants performing different exercises with both correct and incorrect execution.

**Current processed dataset:**

| Property | Value |
|---|---:|
| Participants | 25 |
| Exercises | 2 |
| Bicep Curl videos | 150 |
| Squat videos | 150 |
| Total videos | 300 |
| Good samples | 150 |
| Bad samples | 150 |
| Camera viewpoints | 3 |

### Supported Exercises

- Bicep Curl
- Squat

### Camera Viewpoints

- Front
- Side
- Diagonal

The dataset is organized by participant, exercise, execution quality, and camera viewpoint.

*Note: The original dataset README reports 26 participants, while the current processed subset contains videos from 25 participants.*

## ⚙️ Project Workflow

The system follows this general pipeline:

1. **Video Input:** Load exercise videos from the dataset.
2. **Video Processing:** Extract frames using OpenCV.
3. **Pose Estimation:** Detect human body landmarks using MediaPipe.
4. **Feature Extraction:** Convert detected landmarks into numerical features.
5. **Data Preparation:** Organize the extracted features and labels.
6. **Model Training:** Train machine learning models using the processed data.
7. **Prediction:** Classify exercise execution as Good or Bad.
8. **Evaluation:** Analyze accuracy, precision, recall, F1-score, confusion matrices, and classification errors.

## 🧠 Machine Learning Approach

### 1. Pose Estimation and Feature Extraction

MediaPipe is used to extract body pose landmarks from each video frame.

The temporal feature dataset currently contains:

- 300 processed videos.
- 60 sampled frames per video.
- 33 pose landmarks per frame.
- 4 values per landmark.
- 132 features per frame.

The resulting temporal dataset has the shape:

`(300, 60, 132)`

### 2. Baseline Model — Random Forest

Random Forest was implemented as the baseline machine learning classifier.

The model uses extracted pose-based features to classify exercise execution into two categories:

- Good
- Bad

A participant-level data split was used to reduce data leakage between training, validation, and testing.

### 3. Advanced Models — In Development

The project explores temporal deep-learning approaches that can learn movement patterns across consecutive video frames.

Planned and prepared architectures include:

- Temporal CNN
- LSTM
- GRU
- Transformer
- ST-GCN

These models are intended for future experimentation and comparison. Their performance will be reported after training and evaluation.

## 📊 Experimental Results

### Random Forest — Validation Results

Average performance across three random seeds:

| Metric | Score |
|---|---:|
| Accuracy | 67.59% |
| Macro Precision | 68.81% |
| Macro Recall | 67.59% |
| Macro F1-score | 67.11% |

### Random Forest — Test Results

| Metric | Score |
|---|---:|
| Accuracy | 58.33% |
| Macro Precision | 58.42% |
| Macro Recall | 58.33% |
| Macro F1-score | 58.23% |

**Test predictions:**
- Correct predictions: 35
- Incorrect predictions: 25
- Total test videos: 60

### Performance by Exercise

| Exercise | Test Accuracy |
|---|---:|
| Bicep Curl | 60.00% |
| Squat | 56.67% |

### Performance by Camera View

| Camera View | Test Accuracy |
|---|---:|
| Diagonal | 65.00% |
| Front | 55.00% |
| Side | 55.00% |

These results show that exercise-form classification remains challenging, particularly when distinguishing between correct and incorrect movement patterns. Further experimentation with temporal models and improved feature extraction is planned.

## 🔍 Error Analysis

The Random Forest test evaluation identified several areas for improvement:

- Squat and bicep curl classification both show room for improvement.
- The model struggles particularly with some correctly performed squats.
- Camera viewpoint influences prediction performance.
- Some incorrect predictions occur with relatively high confidence.

Error analysis is used to identify weaknesses in the current feature representation and guide future model development.

## 📈 Evaluation Metrics

The project uses the following evaluation metrics:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Per-exercise accuracy
- Per-camera accuracy
- Prediction confidence
- Error analysis

## 💻 Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/AI-Fitness-Trainer.git
```

Navigate to the project directory:

```bash
cd AI-Fitness-Trainer
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install the required libraries:

```bash
pip install opencv-python mediapipe numpy pandas scikit-learn matplotlib seaborn torch torchvision streamlit
```

Install any additional packages required by your application or ML scripts.

## ▶️ Running the Project

Run the application using the appropriate entry point in your repository.

For a Streamlit application:

```bash
streamlit run streamapp.py
```

For individual Python scripts:

```bash
python filename.py
```

Replace `filename.py` with the actual script you want to execute.

## 🔮 Future Improvements

- Train and evaluate temporal CNN architectures.
- Implement LSTM and GRU models for sequential movement analysis.
- Experiment with Transformer-based architectures.
- Investigate skeleton-based graph neural networks such as ST-GCN.
- Improve pose-based feature engineering.
- Expand the dataset with additional participants and exercises.
- Improve generalization across camera viewpoints.
- Develop real-time exercise-form feedback.
- Provide detailed feedback about detected posture mistakes.
- Compare model performance, inference speed, and computational requirements.

## ⚠️ Current Limitations

- The current processed dataset contains only two exercises.
- The dataset is relatively small.
- Performance varies across exercises and camera viewpoints.
- The current Random Forest model has limited test accuracy.
- Advanced temporal models are still under development.
- The system is an experimental exercise-analysis project and should not be treated as a replacement for professional coaching or medical advice.

## 👨‍💻 Project Focus

This project combines:

- Computer Vision
- Human Pose Estimation
- Machine Learning
- Deep Learning
- Video Processing
- Temporal Sequence Analysis
- Model Evaluation and Error Analysis

## 📚 References

- [MediaPipe](https://developers.google.com/mediapipe)
- [OpenCV](https://opencv.org/)
- [Scikit-learn](https://scikit-learn.org/)
- [PyTorch](https://pytorch.org/)
- [Random Forest — Scikit-learn Documentation](https://scikit-learn.org/stable/modules/ensemble.html#forest)
- [Spatial Temporal Graph Convolutional Networks](https://arxiv.org/abs/1801.07455)

---

**AI Fitness Trainer — Intelligent Exercise Analysis Using Computer Vision and Machine Learning.**
