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

The model is trained with a batch size of 32 and for 100 epochs. Early stopping is applied in this training. The training stops at epoch 24, with a training accuracy of 92% and validation accuracy of 84%. The two figures below show the graph of the training process.

![Loss_graph](https://user-images.githubusercontent.com/95268200/176726631-a83d711e-f3fb-4882-a7fa-8ed0c9bf2be1.PNG)

![Accuracy_graph](https://user-images.githubusercontent.com/95268200/176726627-fa407e79-0ba1-4846-b154-80ef93c61ff0.PNG)

4. Results

Upon evaluating the model with test data, the model obtain the following test results, as shown in figure below.

![test_result](https://user-images.githubusercontent.com/95268200/176726747-81736bd8-6c62-4132-8db3-01ded33b834d.PNG)


