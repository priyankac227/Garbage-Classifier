# ♻️ Garbage Classification Using CNN

A deep learning project that classifies waste images into six categories using a Convolutional Neural Network (CNN). The trained model is integrated with a Streamlit web application, allowing users to upload an image and receive a predicted waste category along with a confidence score.

## 📌 Project Overview

Proper waste classification is an important step toward effective recycling and waste management. This project uses computer vision to automatically classify waste images into the following categories:

- Cardboard
- Glass
- Metal
- Paper
- Plastic
- Trash

The CNN model was trained from scratch using the TrashNet dataset. The project includes dataset preparation, image preprocessing, class imbalance handling, model training, performance evaluation, model checkpointing, and a Streamlit-based prediction interface.

## 🎯 Objectives

- Build a CNN model for multiclass waste image classification.
- Prepare and split the dataset into training, validation, and testing sets.
- Handle class imbalance using class weights.
- Improve training performance using dropout, learning-rate scheduling, and early stopping.
- Evaluate the model using accuracy, precision, recall, F1-score, and a confusion matrix.
- Save the trained model for reuse without retraining.
- Develop an interactive web application using Streamlit.

## 🛠️ Technologies Used

- **Programming Language:** Python
- **Deep Learning:** TensorFlow, Keras
- **Image Processing:** Pillow, NumPy
- **Data Processing:** Python, NumPy
- **Model Evaluation:** Scikit-learn
- **Visualization:** Matplotlib
- **Web Application:** Streamlit
- **Development Environment:** VS Code

## 📂 Dataset

The project uses the TrashNet waste image dataset, containing six waste categories.

The dataset was divided into training, validation, and testing sets.

| Dataset split | Images |
|---|---:|
| Training | 1,766 |
| Validation | 377 |
| Testing | 384 |
| **Total** | **2,527** |

The dataset was split by class to preserve representation across the three subsets.

### Class distribution

| Class | Training | Validation | Testing |
|---|---:|---:|---:|
| Cardboard | 282 | 60 | 61 |
| Glass | 350 | 75 | 76 |
| Metal | 287 | 61 | 62 |
| Paper | 415 | 89 | 90 |
| Plastic | 337 | 72 | 73 |
| Trash | 95 | 20 | 22 |
| **Total** | **1,766** | **377** | **384** |

The dataset is imbalanced, particularly for the trash category. Class weights were used during training to give greater importance to underrepresented classes.

## 🧠 Model Architecture

A custom Convolutional Neural Network was built and trained from scratch.

### Architecture components

1. **Input layer:** RGB images resized to 160 × 160 pixels.
2. **Convolutional blocks:** Convolution layers with 32, 64, 128, and 192 filters to learn visual features.
3. **Pooling layers:** Reduce spatial dimensions and computational requirements.
4. **Regularization:** Dropout and batch normalization to help control overfitting and stabilize training.
5. **Final convolution:** A 1 × 1 convolution with 256 filters.
6. **Global Average Pooling:** Converts feature maps into a compact feature representation.
7. **Dense layer:** 256 units for learning higher-level combinations of features.
8. **Output layer:** Six neurons with softmax activation, one for each waste category.

### Model summary

| Property | Value |
|---|---|
| Model type | Custom CNN |
| Input shape | 160 × 160 × 3 |
| Output classes | 6 |
| Total parameters | 960,358 |
| Trainable parameters | 958,182 |
| Non-trainable parameters | 2,176 |

## ⚙️ Data Preprocessing and Training

The following steps were used in the training pipeline:

- Resize images to 160 × 160 pixels.
- Convert images to RGB format.
- Normalize pixel values to the range 0–1.
- Split the dataset into training, validation, and testing subsets.
- Apply class weights to address class imbalance.
- Train the CNN using a maximum of 35 epochs.
- Monitor validation performance during training.

### Training optimization

The training pipeline uses the following techniques:

- **Dropout:** Helps reduce overfitting.
- **Batch normalization:** Helps stabilize neural network training.
- **Class weighting:** Gives more importance to underrepresented classes.
- **ReduceLROnPlateau:** Reduces the learning rate when validation performance stops improving.
- **Early stopping:** Stops training when the monitored validation metric no longer improves and restores the weights from the best epoch.
- **Model checkpointing:** Saves model weights at the selected best checkpoint.

Training completed with 35 epochs, and early stopping restored the best weights from epoch 29.

## 📊 Model Performance

The model was evaluated on a separate test set containing 384 images.

| Metric | Result |
|---|---:|
| Test Accuracy | 76.30% |
| Macro Precision | 75.06% |
| Macro Recall | 76.85% |
| **Macro F1-score** | **74.67%** |
| Weighted Precision | 79.20% |
| Weighted Recall | 76.30% |
| Weighted F1-score | 77.12% |

### Classification report

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Cardboard | 1.0000 | 0.8525 | 0.9204 |
| Glass | 0.7000 | 0.7368 | 0.7179 |
| Metal | 0.6615 | 0.6935 | 0.6772 |
| Paper | 0.8488 | 0.8111 | 0.8295 |
| Plastic | 0.8644 | 0.6986 | 0.7727 |
| Trash | 0.4286 | 0.8182 | 0.5625 |

### Performance observations

- Cardboard achieved the highest F1-score.
- Paper classification also performed relatively well.
- Trash recall was high, but its precision was comparatively low, indicating more false-positive predictions.
- Glass, metal, and plastic remain areas for improvement.
- The difference between macro and weighted F1-scores highlights the importance of evaluating performance across individual classes, especially with an imbalanced dataset.

These results establish a baseline for future improvements, such as transfer learning, data augmentation, and further model optimization.

## 💾 Saved Models

The project saves the trained models for reuse.

| File | Purpose |
|---|---|
| `trashnet_cnn_from_scratch_final.h5` | Final trained model used by the Streamlit application |
| `trashnet_cnn_from_scratch_best.h5` | Best checkpoint restored during training |

The saved model allows predictions to be made without running the entire training process again.

## 🌐 Streamlit Web Application

The project includes an interactive web application built with Streamlit.

### Application features

- Upload a waste image in JPG, JPEG, or PNG format.
- Preview the uploaded image.
- Classify the image into one of six waste categories.
- Display the predicted class.
- Display the model's confidence score.

The application loads the saved CNN model and preprocesses each uploaded image to match the model's input requirements.

**Note:** The displayed confidence score comes from the model's softmax output and is not necessarily a calibrated probability. The model may also misclassify unfamiliar images or images that do not contain waste.

## 🚀 Installation and Setup

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Garbage-Classifier
```

Replace the placeholder with your repository's actual GitHub URL.

### 2. Create a virtual environment (optional)

```bash
python -m venv venv
```

Activate it on Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install streamlit tensorflow numpy pillow scikit-learn matplotlib
```

Use a TensorFlow version compatible with your Python version and the environment in which the model was trained.

## ▶️ Running the Project

### Train the model

If you want to train the model again, run:

```bash
python train.py
```

This runs the training pipeline and saves the model files specified in the training script.

**You do not need to retrain the model to use the web application** if the saved model already exists.

### Launch the Streamlit application

Make sure `app.py` and `trashnet_cnn_from_scratch_final.h5` are in the project directory, then run:

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal, usually:

`http://localhost:8501`

Upload an image and view the predicted waste category and confidence score.

## 📁 Project Structure

```text
Garbage-Classifier/
│
├── train.py
├── app.py
├── trashnet_cnn_from_scratch_final.h5
├── trashnet_cnn_from_scratch_best.h5
│
├── trashnet_split_optimized/
│   ├── train/
│   ├── val/
│   └── test/
│
├── dataset-resized/
├── dataset-resized.zip
├── requirements.txt
└── README.md
```

*The structure above represents the main project files and dataset folders. Your exact directory structure may differ depending on your training script and local setup.*

## 🔮 Future Improvements

- Apply transfer learning using MobileNetV2, EfficientNet, or another pretrained CNN.
- Experiment with data augmentation to improve generalization.
- Improve precision for the trash category.
- Analyze misclassifications using a confusion matrix.
- Compare models using macro F1-score and per-class recall.
- Tune hyperparameters systematically.
- Calibrate prediction confidence.
- Deploy the application to a cloud hosting platform.

## 📌 Limitations

- The model was trained on a relatively small dataset.
- Performance may vary with lighting, backgrounds, image quality, and camera angles.
- The model is limited to six predefined classes.
- A prediction is not guaranteed to be correct, even when its confidence score is high.
- The reported test metrics reflect the current trained model and may differ after retraining.

## 👩‍💻 Author

**Priyanka Choudhary**

---

*This project demonstrates an end-to-end computer vision workflow, from dataset preparation and CNN training to model evaluation and deployment through an interactive Streamlit application.*
