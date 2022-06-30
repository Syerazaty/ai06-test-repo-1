Heart Disease Prediction Using Feedforward Neural Network

1. Summary

The aim of this project is to create a highly accurate deep learnning model to predict whether the patient has heart disease or not.

2. IDE and Framework

This project is created using Sypder as the main IDE. The main frameworks used in this project are Pandas, Scikit-learn and TensorFlow Keras.

3. Methodology

3.1. Data Pipeline
The data is first loaded and preprocessed, such that unwanted features are removed, and label is encoded in one-hot format. Then the data is split into train-validation-test sets, with a ratio of 60:20:20.

3.2 Model Pipeline

A feedforward neural network is constructed that is catered for classification problem. The structure of the model is fairly simple. Figure below shows the structure of the model.

![model](https://user-images.githubusercontent.com/95268200/176714750-1a724051-dd66-4919-bae1-59d98c2b7a00.png)

The model is trained with a batch size of 32 and for 100 epochs. Early stopping is applied in this training. The training stops at epoch 24, with a training accuracy of 99% and validation accuracy of 95%. The two figures below show the graph of the training process.

![Loss_graph](https://user-images.githubusercontent.com/95268200/176714803-03aa7574-08fd-44a7-bb78-503f1f082779.PNG)

![Accuracy_graph](https://user-images.githubusercontent.com/95268200/176714831-5d8a75b0-3f4c-451c-a7fc-b618a3d72c1f.PNG)

4. Results

Upon evaluating the model with test data, the model obtain the following test results, as shown in figure below.

![test_result](https://user-images.githubusercontent.com/95268200/176714872-72b93fef-17f4-46d8-84ba-87ff24079587.PNG)

