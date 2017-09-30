# Face_expression_Recognizer_CNN

This project uses convolutional neural networks to recognize the facial expression of the face. The system recognizes 
7 facial expression, viz. 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral.
HAAR Frontal face detection has been used through openCV to capture image from Webcam and detect face.
The detected face is then fed into the CNN for expression recognition. The architecture of the CNN model has
2 Convolutional layer of 32 filters,2 Convolutional layer of 32 filters,2 Convolutional layer of 64 filters, 2 Convolutional layer of 64 filters, MaxPool layer of (2,2), 2 Convolutional layer of 128 filters,MaxPool layer of (2,2),
and then Flatten, Dense and Dropout layers connected. 
The dataset used for training purpose is fer2013 dataset and can be found on kaggle.com

www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data
