<h2 align="center">Emotion and Gender Recognition for Human-Computer Interaction</h2>

<p align="center"><i>A real-time facial analysis system combining CNN-based emotion recognition with SVM-based gender classification</i></p>

<div align="center">This project implements a dual-model approach for real-time facial analysis, achieving <b>67% accuracy</b> in emotion detection and <b>88% accuracy</b> in gender classification.</div>

<h2 align="center">Features</h2>

- Real-time emotion recognition using Convolutional Neural Networks (CNN)
- Gender classification using Support Vector Machine (SVM) with HOG features
- Support for multiple emotion classes including: Happy, Sad, Neutral, Angry, and Fear
- Real-time processing with frame smoothing (10-frame window)
- Gaussian blur and Haar cascade implementation for improved face detection
- Batch normalization and dropout layers to prevent overfitting

<h2 align="center">Model Architecture</h2>

<table>
<tr>
<td width="50%">

### Emotion Classification (CNN)
- 4 Convolutional blocks including:
  - Convolution Layer
  - Max Pooling Layer
  - Batch Normalization
  - Dropout Layer
- Fully Connected blocks with:
  - Flatten Layer
  - Dense Layer
  - Batch Normalization
  - Final Dense Layer with Softmax activation

</td>
<td width="50%">

### Gender Classification (SVM)
- HOG (Histogram of Oriented Gradients) feature extraction
- Linear kernel SVM classifier
- 80-20 train-test split ratio
- Feature preprocessing and normalization
- Optimized hyperparameter selection

</td>
</tr>
</table>

<h2 align="center">Performance Metrics</h2>

<table>
<tr>
<td width="50%">

### Emotion Recognition Results
- Test Accuracy: 0.67
- Test Loss: 1.1542
- F1-score: 0.65

</td>
<td width="50%">

### Gender Recognition Results
- Accuracy: 0.88
- Female F1-score: 0.88
- Male F1-score: 0.88

</td>
</tr>
</table>

<h2 align="center">Installation Instructions</h2>

### Prerequisites
- Python 3.7 or higher
- Git
- pip (Python package installer)

### Required Model Download
Before running the project, you must download the pre-trained model:
1. Download the model file from [this Google Drive link](https://drive.google.com/file/d/1dTc5i03YORHfJlixI6TDWCmLmEsLin2O/view?usp=sharing)
2. Create a folder named `models` in your project root directory
3. Place the downloaded file in the `models` folder

> **Note**: The model file exceeds GitHub's 100MB file size limit and is therefore not included in the repository. You must download it separately.

### Windows Installation

1. Install Python:
   ```bash
   # Download Python from the official website
   https://www.python.org/downloads/windows/
   # During installation, make sure to check "Add Python to PATH"
   ```

2. Install Git:
   ```bash
   # Download Git from
   https://git-scm.com/download/windows
   ```

3. Clone the repository:
   ```bash
   git clone [your-repository-url]
   cd [repository-name]
   ```

4. Create and activate virtual environment:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

5. Install PyTorch:
   ```bash
   # CPU only
   pip3 install torch torchvision torchaudio
   
   # With CUDA support (if you have NVIDIA GPU)
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

6. Install other dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### macOS Installation

1. Install Homebrew (if not already installed):
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"
   ```

2. Install Python:
   ```bash
   brew install python
   ```

3. Install Git:
   ```bash
   brew install git
   ```

4. Clone the repository:
   ```bash
   git clone [your-repository-url]
   cd [repository-name]
   ```

5. Create and activate virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

6. Install PyTorch:
   ```bash
   # For both Intel and Apple Silicon Macs
   pip3 install torch torchvision torchaudio
   ```

7. Install other dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Verification
To verify the installation, run:
```bash
python
>>> import torch
>>> print(torch.__version__)
>>> exit()
```

<h2 align="center">Dataset</h2>

This project uses:
- FER 2013 Dataset for emotion recognition
- Gender Dataset from Kaggle for gender classification

<h2 align="center">Limitations</h2>

- Accuracy variations across different emotions
- Limited by image size in the dataset
- Grayscale image processing only
- Performance impact with multiple faces in frame
- Real-time processing stutters with multiple subjects

<h2 align="center">Future Improvements</h2>

- Enhanced accuracy through additional data augmentation
- Optimization for multiple face detection
- Model architecture refinements
- Performance optimization for real-time processing

<h2 align="center">References</h2>

- [FER 2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013/data)
- [Gender Classification Dataset](https://www.kaggle.com/datasets/yasserhessein/gender-dataset)
- [A study on computer vision for facial emotion recognition](https://www.nature.com/articles/s41598-023-35446-4.pdf)
- [Gender classification with support vector machines](https://ieeexplore.ieee.org/document/840651)
- [Real Time CNN for Emotion and Gender Classification](https://github.com/oarriaga/face_classification/blob/master/report.pdf)
