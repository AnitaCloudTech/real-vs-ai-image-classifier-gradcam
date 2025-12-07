# 🧠 Real vs AI Image Classifier + Grad-CAM

This project uses a **Convolutional Neural Network (CNN)** to classify images as **real** or **AI-generated**, with visual explanations of the model's decisions using the **Grad-CAM** technique.  
It provides a simple **Gradio interface** where users can upload an image and get both the prediction and a heatmap overlay.

---

## 🔥 Key Features
- Real vs AI image classification  
- CNN implemented in TensorFlow/Keras  
- Automatic detection of the last Conv2D layer for Grad-CAM  
- Visual explanation of important regions in the image (Grad-CAM overlay)  
- Deployable via Gradio interface  
- Fully extendable for further training or improvements  

---

## 📁 Project Structure

```plaintext
project/
│
├── dataset/ # Folder with subfolders for each class
│ ├── real/
│ └── ai/
│
├── Real_vs_AI_Classifier_GradCAM_Gradio.ipynb # Main Jupyter Notebook
├── requirements.txt # Dependencies
└── README.md

---

## 🛠️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/AnitaCloudTech/real-vs-ai-gradcam.git
cd real-vs-ai-gradcam
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### Or manually:
```bash
pip install tensorflow==2.19.0 matplotlib gradio tensorflow-datasets torch torchvision \
diffusers transformers accelerate safetensors
```
### 🧩 Dataset

The dataset folder should have the following structure:
```plaintext
dataset/
 ├── real/      # real-world images
 └── ai/        # AI-generated images


Folder names are automatically used as labels for training.

### 🧠 Model Architecture

Simple CNN:

Conv2D (16 filters) → MaxPooling
Conv2D (32 filters) → MaxPooling
Conv2D (64 filters)
GlobalAveragePooling
Dense(NUM_CLASSES, softmax)


Optimizer: Adam

Loss: categorical_crossentropy

Output activation: softmax

### 🔍 Grad-CAM

Automatically detects the last Conv2D layer

Generates a heatmap highlighting important image regions

Overlay on the original image for visual explanation

Returned in the Gradio interface alongside predictions

### 🚀 Running the Notebook

Open the notebook in Jupyter Lab/Notebook or Google Colab:

```bash
jupyter notebook Real_vs_AI_Classifier_GradCAM_Gradio.ipynb

```

Run all cells sequentially.

Interact with the Gradio interface inside the notebook:

Upload an image

Get predicted class (Real / AI)

See Grad-CAM overlay highlighting key areas

### 🔮 Ideas for Improvement

Use stronger CNN backbones like ResNet or EfficientNet

Apply data augmentation for better generalization

Multi-channel Grad-CAM or LIME/SHAP explainability

Export trained model to TensorFlow Lite for mobile use

UI with batch prediction for multiple images
