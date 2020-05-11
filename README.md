# Face-Recognition-and-Sentiment-Detection

This application uses OpenCv to identify and recognize faces from a given set of training data. It uses several libraires as numpy, pickle etc.

How does it work?

OpenCv uses the webcam to scan your face and find the Region of Interest(ROI). Numpy then converts the given images(Dataset) into integer matrix that contains integral values for every faces. Then this is matched with the face that is currently appearing in the webcam.
It then checks the resemblance of the image with given dataset's images and gives the most appropriate outcome based on the data.

Sentiment analysis will be added henceforth
