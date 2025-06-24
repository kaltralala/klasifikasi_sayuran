# 🥬 VegNet: Custom CNN for 15-Class Vegetable Image Classification

## 📌 Overview
**VegNet** is a deep learning project that classifies 15 types of vegetables using a custom-built Convolutional Neural Network (CNN). This project was completed as part of the final assignment in the “[Belajar Fundamental Deep Learning](https://www.dicoding.com/certificates/81P2L9YKYZOY)” course by [Dicoding](https://www.dicoding.com), and serves as a personal showcase of my skills in computer vision, model interpretability, and CNN architecture design.

The final model achieves:
- ✅ **Training Accuracy:** 88.02%  
- ✅ **Validation Accuracy:** 93.33%  
- ✅ **Test Accuracy:** 91.78%

This project emphasizes understanding the internals of CNNs without relying on pre-trained models. All classification results are based on my own architecture and data processing pipeline.

---

## 🧠 Project Goals
- Build a high-performing image classifier without using pre-trained models
- Practice data augmentation and manual data splitting
- Gain hands-on experience with model interpretability and visualization
- Package the entire training and evaluation process in a single Jupyter notebook

---

## 🗂 Dataset
- **Source:** [Vegetable Image Dataset – Kaggle](https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset)
- **Total Images:** 21,000 images (15 classes × 1,400 images each)
- **Image Size:** 224×224 pixels (JPG format)
- **Classes:**  
  `Bean, Bitter Gourd, Bottle Gourd, Brinjal, Broccoli, Cabbage, Capsicum, Carrot, Cauliflower, Cucumber, Papaya, Potato, Pumpkin, Radish, Tomato`

**Dataset Preparation:**
- Only the original `train/` folder was used.
- Custom script was created to split into new folders: `train/`, `val/`, and `test/` (70/15/15).
- Image augmentation and preprocessing handled using `ImageDataGenerator`.

---

## 🧪 Preprocessing & Augmentation

All input images were normalized by scaling the pixel values to a [0, 1] range.

To improve the model’s generalization ability, especially under real-world variations, several augmentation techniques were applied to the training data:

- **Random rotation up to 40 degrees**: Helps the model become invariant to orientation changes.
- **Horizontal and vertical shifts up to 20%**: Makes the model robust to positional variance of objects in the image.
- **Shear transformations of 20%**: Introduces affine distortion to simulate perspective changes.
- **Zoom augmentation up to 30%**: Encourages the model to learn features at various scales.
- **Horizontal flipping**: Adds mirrored versions of each image to expand spatial diversity.
- **Filling empty pixels using 'nearest' strategy**: Ensures smooth pixel values after transformations.

Only the training set was augmented in this way.  
The **validation and test sets** were kept clean, with only pixel rescaling applied. This allows for unbiased performance evaluation on real, unmodified samples.

---

## 🧱 Model Architecture

The model was built from scratch using the Sequential API in Keras, without using any pre-trained architectures. The architecture is composed of four convolutional blocks followed by fully connected layers, structured as follows:

Conv2D(32, kernel_size=3) → MaxPooling2D

Conv2D(64, kernel_size=3) → MaxPooling2D

Conv2D(128, kernel_size=3) → MaxPooling2D

Conv2D(128, kernel_size=3) → MaxPooling2D

→ Flatten → Dense(512) → Dropout(0.5) → Dense(15, softmax)

| Layer Type          | Output Shape       | Parameters |
|---------------------|--------------------|------------|
| Conv2D (32 filters) | (222, 222, 32)     | 896        |
| MaxPooling2D        | (111, 111, 32)     | 0          |
| Conv2D (64 filters) | (109, 109, 64)     | 18,496     |
| MaxPooling2D        | (54, 54, 64)       | 0          |
| Conv2D (128 filters)| (52, 52, 128)      | 73,856     |
| MaxPooling2D        | (26, 26, 128)      | 0          |
| Conv2D (128 filters)| (24, 24, 128)      | 147,584    |
| MaxPooling2D        | (12, 12, 128)      | 0          |
| Flatten             | (18432)            | 0          |
| Dense (512 units)   | (512)              | 9,437,696  |
| Dropout (0.5)       | (512)              | 0          |
| Dense (15 classes)  | (15)               | 7,695      |

Total Parameters: **9,686,223**

> The model was designed to be compact yet expressive enough to capture the variations across 15 different vegetable classes. A dropout layer was included to reduce overfitting and enhance generalization.

---

## 📊 Performance & Results

| Metric              | Score     |
|---------------------|-----------|
| Training Accuracy   | 88.02%    |
| Validation Accuracy | 93.33%    |
| Test Accuracy       | 91.78%    |

---

## 📉 Confusion Matrix

Below is the confusion matrix based on the test set predictions.  
All predictions shown are true positives — no class confusion occurred.

![Confusion Matrix](outputs/confusion_matrix.png)

---

## 🖼️ Sample Predictions (True Positives Only)

| Class     | Prediction | Sample |
|-----------|------------|--------|
| Tomato    | Tomato     | ![](outputs/true_predictions/tomato_01.jpg) |
| Cucumber  | Cucumber   | ![](outputs/true_predictions/cucumber_04.jpg) |
| Brinjal   | Brinjal    | ![](outputs/true_predictions/brinjal_09.jpg) |

---

## 🧰 Tools & Technologies
- Python
- TensorFlow & Keras
- NumPy, Pandas
- Matplotlib, Seaborn
- Google Colab

## 🙏 Acknowledgments

This project was completed as part of the final submission for the **[Dicoding course: Belajar Fundamental Deep Learning](https://www.dicoding.com/certificates/81P2L9YKYZOY)**.  
Special thanks to:

- **[Dicoding Indonesia](https://www.dicoding.com)**, for providing high-quality learning materials that guided the foundational understanding of deep learning.
- **Kaggle contributor @misrakahmed**, for making the [Vegetable Image Dataset](https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset) publicly available.
- **@ndiekema**, whose [GitHub project](https://github.com/ndiekema/vegetable_classifier) served as a source of inspiration and reference.
- The vibrant deep learning community for countless open resources, documentation, and motivation.

> This project reflects a learning journey and is meant to showcase model building, interpretability, and experimentation—not to replace industrial-grade solutions.
