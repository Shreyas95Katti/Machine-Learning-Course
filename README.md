# Machine-Learning-Course
Hello Guys!!

## Iris Data Visualization Project
The Iris Project is a data visualization project. It involves the most basic data visualization function. Being beginner freindly I would suggest my fellow Machine Learning enthusiasts to have a look at this. Data visualization is considered to be the first step into any machine learning problem statement. It allows you to view the positioning of the data points and thus hints you the best model for any given problem statement. The given project invloves concepts from two main python libraries Matplotlib and Seborn. It gives you an overview of the few main data visualization functions which are widely used namely: 1)PairPlot 2)Histogram 3)ViolinPlot 4)BoxPlot 5)ScatterMatrix 6)Heatmap

These plots will surely give you a clear cut idea about the model to be used in your machine learning project. It will familiarize you with the data visualizaton techniques!

Dataset: https://www.kaggle.com/datasets/uciml/iris

The dataset contains the following data about the Iris flower 1)ID 2)SepalLength(cm) 3)SepalWidth(cm) 4)PetalLength(cm) 5)PetalWidth(cm) 6)Species


## Boston House Price Prediction Project
The Boston house price prediction project is a highly intutive project. It allows you to explore basic and important machine learning concepts such as Data Visualization, Linear Regression and Robust Regreesion. These topics are the stepping stones to get into core machine learning projects. In the project we focus on single variable linear regression model. This is a supervised learning technique considering that we have a labeled dataset. The project starts with importing the required python libraries and and loading the dataset with appropriate changes. Then we visualize the dataset to observe the data points. This is done using two major data visualization techniques that are pairplot and heatmap. The next step involved is the usage of the linear regression model on the training and testing sub-dataset. We view how the linear regression predicts an accurate result by plotting the best fit curve which is done using the data visualization functions regplot and jointplot. The model is evaluated. Finally we repeat the same procedure for robust regression also. With the knowledge about this dataset we finish the project by visualizing how a perfect dataset and a perfect linear regression model look.

The major take away from this project is: Machine Learning is only 5 steps

1)Import required libraries

2)Choose a model and apply necessary parameters

3)Arrange data into feature matrix and target matrix

4)Fit model fo give data

5)Apply for new data

Dataset: https://www.kaggle.com/code/prasadperera/the-boston-housing-dataset


## Types of Regression
### Multi-Variable Based Regression
Exploring the aforementioned regression model in further depth. Prior to this, the prediction was only dependent on one of the feature matrix parameters. We broaden our vision to focus on the entire dataset in order to develop a generalised model that predicts home prices and indicates the association between the feature matrix and the target variable. This form of regression computes the independent correlation between each parameter on the target variable and adjusts the internal synaptic weights accordingly. The model is trained in such a manner that it recognises particular patterns in the input characteristics and utilises these pattern sets to determine which closely fits the test input and so predicts the price with more accuracy and precision. We next use Eigen Values and Eigen vectors produced from the correlation matrix to test for Colinearity. This test determines if two or more features are exactly correlated.

### Polynomial Regression
Linear regression is a fantastic model for prediction that works flawlessly on the majority of datasets. However, not all data relationships follow a linear pattern. Consider the plot between the columns 'DIS' and 'NOX' in the same Boston dataset. It is clear that this relationship is not linear. We choose to modify the Linear Regression model and transform it into a higher order regression model for such applications. This is possible with SciKit Learn's PreProcessing library's PolynomialFeatures function. This function allows us to choose the order of the needed functions, making them more useable and regularised. We utilised Linear Regression and Polynomial Regression with degrees 2 and 3 for the preceding plot. The outcomes are compared to the R2 score measurements. This issue demonstrates how data visualization makes predicting which model would best portray the data easier.

### Non-Linear Regression
This regression model came into being as a result of a similar given challenge as the preceding one. This concept seeks to handle the problem of data that does not follow a linear pattern. We examine the plot between the columns 'LSTAT' and 'MEDV' to better grasp the issue. The plot appears to be widely dispersed, with a large variation and no clear linear relationship. We apply a few nonlinear regression approaches to this figure, including the Decision Tree Regressor, Random Forest Regressor, and Ada Boost Regressor. The Mean Squared Error and R2 score measurements are used to compare the findings. Then, using the non-linear models described above, we determine which feature from the feature matrix is the most dominating. This will assist us in understanding what the model is interpreting from the data input. As a result, advice on which non-linear model would be the best match for the dataset.

### Regularized Regression
The preceding models concentrate on pattern identification and feature extraction from the dataset. Assume we are working on a dataset but do not know its source or origin. It might result in the model being overfitted. This means that the models utilised learned very well for the data supplied during training but failed to perform well for fresh test data. In layman's words, the model is based on the input output mapping features and so finds it challenging to execute the same with unknown data. The regurlarized regression method is used to tackle such overfitting problems. This approach tries to lower the coefficient of dependancy, thereby eliminating the overfitting problem. For the aforementioned problem, models such as Ridge Regression, Lasso Regression, and Elastic-Net Regression are used. Based on the results, we may conclude that (i) Ridge Regression either contains or excludes the constant. (ii) Lasso Regression achieves coefficient shrinkage but not parameter selection. (iii) Elastic Regression performs best when the covariance is highly correlated.


## Logistic Regression
This is one of the first models I have explored for classification based problems. This model is based on the Rosenblatt's Artificial Neural Network that uses the sigmoid function as its activation function. We first plot out this function to understand what the model intends to do. To understand this better, we create a small dataset to test and observe how this model responds. I further used this model to work on mini projects and for hackathon statements. 


## Cross Validation
Cross Validation is a technique used to train the model better in cases where the dataset size is considerably small. In this method the training dataset is further split into validation set. Validation set is fed into the model and the generated result hints on the modifications that could be done on the model which could possibly make it better. We work on a few cross validation techniques such as the K-fold and The Stratified K-fold technique. We generate the accuracy and cross validation score for a simple dataset and observe how cross validation works
