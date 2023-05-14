# Breast-Cancer
The primary purpose of this project is to create a Machine Learning Model to distinguish benign from malignant tumors for Breast Cancer diagnosis using the Breast Cancer Wisconsin (Diagnostic) Data Set. As such, I will use several different techniques for binary classification and test to see which is most effective.
Neural Networks are the most common and effective tool to develop this type of model. However, they require a large amount of data to train the model, and the dataset we are dealing with only has 569 observations. Thus, I don't believe we have enough data to effectively train the model, so I will instead work with other methods of binary classification. These will each be stored in their own file, labeled by which method of binary classification was used. You should notice that the code itself doesn't change much for each file, mostly because each of these methods require the same data preprocessing and cleaning.

Link to data: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data?resource=download

Results: (averages across 30,000 random states)
SVC with a RBF kernal: 98.63% train accuracy, 97.35% test accuracy
Logistic Regression: 98.94% train accuracy, 97.72% test accuracy
