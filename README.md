# Logistic-Regression-Handling-missing-data-problem
In this repository, the Logistic Regression model first fills the missing data in columns with the mean of the column data for the machine to utilize the data and we have also performed standardization on the data to avoid biases and make the model more accurate to predict true results.


We are here using the full data as both training and test data but we can also split the data into training and test data with the help of library functions so that we train our model on lets say 70% data and test it on the rest 30% data or you can use some new dataset to test.


We are in this problem predicting which coronavirus patients might require intensive care units based on the data available here. But also data has many missing value which we are filling by the median column data. We are also standardizing the data to avoid one feature be dominant and lead to biasing. If you split data into training and testing data, you must standardize both the training and testing data.
