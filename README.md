# ML-Project
Handwritten Digit Classification using K-Nearest Neighbors (KNN)

Project Description:
The objective of this project was to apply the K-Nearest Neighbors (KNN) classification algorithm on the Digits dataset, a popular dataset containing images of handwritten digits.
The dataset consisted of 8x8 pixel grayscale images, where each image represented a digit from 0 to 9. Each image was converted into a 64-dimensional feature vector, with each dimension corresponding to the grayscale intensity of a pixel.

The key steps of the project were as follows:
Data Loading and Exploration:
I loaded the Digits dataset from Scikit-learn’s datasets module and explored its structure, feature dimensions, and target labels.
Sample images from the dataset were visualized using Matplotlib to gain a better understanding of the digit patterns.

Data Splitting:
I split the dataset into training and test sets using train_test_split to ensure unbiased model evaluation.

Model Building and Training:
I implemented the K-Nearest Neighbors (KNN) algorithm using sklearn.neighbors.KNeighborsClassifier.
The model was trained on the training dataset with an initial choice of k (number of neighbors), and later optimized based on accuracy.

Model Evaluation:
The trained model was evaluated on the test set.
I calculated key performance metrics such as:
Accuracy Score
Confusion Matrix
Classification Report (Precision, Recall, F1-score)

I also visualized the confusion matrix to interpret the model’s performance across different digit classes.
Hyperparameter Tuning:
To improve performance, I experimented with different values of k and selected the optimal k based on validation accuracy.
Overall Outcome:
The project successfully demonstrated the application of KNN classification for image recognition tasks. It provided hands-on experience in supervised learning, model evaluation, and hyperparameter tuning using Scikit-learn and Python visualization tools.

