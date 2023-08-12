# Real-Time-Face-Mask-Detection
Second task submission at the Sync's Intern

The real time face mask detection project is a computer vision task that uses deep learning techcniques to detect wetger a person is wearing a facemask or not.

THE PROJECT INCLUDES THE FOLLOWING KEY FEATURES:

Face detection: Utilizing a pre-trained face detection model, faces are detected within the input frame.

Mask prediction: Applying a trained deep learning model, the system predicts whether each face is wearing a mask or not.

Real-time processing: The system operates in real-time, allowing for immediate feedback on mask detection.

This process involves a combination of Machine Learning, Image Processing and pattern recognition techniques to achieve accurate results.
For my real time detection i used Python, Tensorflow, Opencv and a pre-trained MobileNetV2 model.

Process Walkthrough:

* Data collection: The training dataset (from Kaggle.com) consisted of images labeled as "with mask" and "without mask."
  
* Preprocessing: Preprocessed the images to ensure they are of consistent size and format. Also, used one hot encoding to create corresponding labels for each image.
  
* Splitting the Dataset: Divided the dataset into training, validation, and testing sets. The training set will be used to train the model, the validation set to tune hyperparameters, and the testing set to evaluate the model's final performance.
  
* Choosing a deep learning network framework: TensorFlow is the deep learning network framework used for this project
  
* Data Augmentation: To enhance model performance and prevent overfitting, data augmentation techniques such as rotation, flipping, and zooming was done during training.
  
* Building the Model: For this task, MobileNetV2, a pre-trained Convolutional Neural Network (CNN) was used as a feature extractor and some fully connected layers were aded on top and fine tuned. This is process is called Transfer Learning.
  
* Compilling and Training the Model: The model was compiled with an appropriate loss function (binary cross-entropy) and an optimizer (Adam). The model was trained using the training data and validated using the validation data. The process was monitored for overfitting.
  
* Evaluating the Model: To measure the model's performance. Evaluations include accuracy, precision, recall, and F1-score.
  
* Save the Model: After satisfaction with the model's performance, It was ready for use in real-time.
  
* Real-time Face Detection: For the real-time face detection, OpenCV was used as the face detection library to identify and extract faces from a video stream. The necessary libaries were imported, the pre-trained model was loaded, the face detection model was loaded, the functions for the face mask detection were defined, the real time fask mask detection webcam was activated and running the main() fuction started the live stream for the face mask detection.

RESULT: The face mask detection system demonstrated impressive accuracy and efficiency in detecting wether a face mask was worn or not via the real time video stream.





