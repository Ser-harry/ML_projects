# Image Classification using K-Nearest Neighbors (KNN) in CV Studio

Welcome to the **K-Nearest Neighbors (KNN) Image Classification** project! This project demonstrates how to load a pre-trained KNN model to classify images using the CV Studio tool by IBM. You will be able to upload images, process them, and classify them based on previously trained data using KNN.

## Objectives

The main objectives of this project are:

1. **Load the saved KNN model**: Retrieve the KNN model that was previously trained.
2. **Upload an image**: Upload a new image for classification.
3. **Classify the image**: Process the image and classify it using the loaded KNN model.

By following the steps outlined below, you will be able to successfully classify images and gain insight into how the KNN algorithm works in practice.

## Table of Contents

- [Getting Started](#getting-started)
- [Prerequisites](#prerequisites)
- [Project Workflow](#project-workflow)
    - [1. Load Important Libraries](#1-load-important-libraries)
    - [2. Setup CV Studio Client](#2-setup-cv-studio-client)
    - [3. Load the Saved KNN Model](#3-load-the-saved-knn-model)
    - [4. Upload and Process Your Image](#4-upload-and-process-your-image)
    - [5. Classify the Image](#5-classify-the-image)
- [Results](#results)
- [Authors](#authors)
- [License](#license)

---

## Getting Started

To get started with this project, you will need to set up the environment, retrieve your saved KNN model, and upload the image you wish to classify. This project assumes you have already trained a KNN model and stored it in **CV Studio**.

## Prerequisites

Make sure you have the following libraries installed before starting the project:

- `numpy`
- `matplotlib`
- `opencv-python`
- `skillsnetwork`

You can install the missing libraries using pip:

```bash
pip install numpy matplotlib opencv-python skillsnetwork
```

---

## Project Workflow

### 1. Load Important Libraries

We will start by importing the necessary libraries for data processing, image visualization, and image classification.

```python
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skillsnetwork import cvstudio
```

### 2. Setup CV Studio Client

Next, initialize the CV Studio client to access your project files and retrieve the saved KNN model and annotations.

```python
cvstudioClient = cvstudio.CVStudio()
annotations = cvstudioClient.get_annotations()
```

### 3. Load the Saved KNN Model

Retrieve the previously trained KNN model and its details from CV Studio. This includes fetching the model parameters such as the optimal number of neighbors (k_best) and the training data.

```python
model_details = cvstudioClient.downloadModel()
k_best = model_details["k_best"]

fs = cv2.FileStorage(model_details['filename'], cv2.FILE_STORAGE_READ)
knn_yml = fs.getNode('opencv_ml_knn')
samples = knn_yml.getNode('samples').mat()
responses = knn_yml.getNode('responses').mat()
fs.release()

knn = cv2.ml.KNearest_create()
knn.train(samples, cv2.ml.ROW_SAMPLE, responses)
```

### 4. Upload and Process Your Image

You will upload an image from your local system, convert it to grayscale for simplicity, and resize it to match the input size required by the KNN model.

```python
my_image = cv2.imread("your_uploaded_file.jpg")
my_image = cv2.cvtColor(my_image, cv2.COLOR_BGR2GRAY)
my_image = cv2.resize(my_image, (32, 32))

# Flatten the image into a numpy array
pixel_image = my_image.flatten()
pixel_image = np.array([pixel_image]).astype('float32')
```

### 5. Classify the Image

Using the pre-loaded KNN model, classify the uploaded image based on the nearest neighbors and display the classification result.

```python
ret, result, neighbours, dist = knn.findNearest(pixel_image, k=k_best)
print(neighbours)
print('Your image was classified as a ' + str(annotations['labels'][int(ret)]))
```

---

## Results

After running the classification process, you will see the predicted class of your image. The KNN model classifies the image based on the k nearest neighbors from the training data, and the result will be shown as the majority class among these neighbors.

For example:
```
[[0.]]
Your image was classified as a dog
```

---

## Authors

This project was created by **Aije Egwaikhide**, a Data Scientist at IBM. Aije holds a degree in Economics and Statistics from the University of Manitoba and a Post-grad in Business Analytics from St. Lawrence College, Kingston. She is currently pursuing her Masters in Management Analytics at Queens University.

---

## License

This project and all related materials are © IBM Corporation 2021. All rights reserved.

---

Thank you for using this tutorial! You can revisit the **CV Studio** tool at any time to explore your saved projects and models.

