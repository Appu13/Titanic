## Introductions
This is my submission for the kaggle titanic problem. You can check it out [here](https://www.kaggle.com/c/titanic/overview). The problem is well documented and has a lot of solutions but here I try to implement my take on the problem. Hope you like it. XD

## Data analysis and cleaning

At the start we import the dataset and the needed libraries. After inital viewing of the shape, info and description of the data we find that we have missing elements in the Age, Embarked and Cabin columns so we need to perform some manipulation of the data

### Age
On the initalize analysis of this dataset we find there are 3 sets of people adult(male or names with title as Mr), adult(female or names with title as Mrs) and children(names with title as Miss, master etc) we cannot just apply the mean age of the data as this will lead to data corruption so we split the data into distinct categories and fill the missing values for each age and then remerge them. The code for one of the female adults is below the rest follow a similar so I am not including them

```{python}
#dealing with missing values for age
mrs_data = data.loc[(data["Sex"] == "female") & (data.Name.str.contains("Mrs"))]
# mrs_data.describe()

# Visualizing the age distribution
#mrs_data.Age.hist()
#mrs_data.Age.plot(kind = "kde")

#Based on the distribution we can choose the mean as a measure of our central tendency
mrs_data['Age'].fillna(int(mrs_data['Age'].mean()),inplace = True)
mrs_data.info()
```

Finally we merge all the separate tables together
```{python}
common = data.merge(sdata, on=["PassengerId"])
result = data[~data.PassengerId.isin(common.PassengerId)]
#result.Age.plot(kind = "kde")

result['Age'].fillna(int(result['Age'].mean()), inplace = True)
result.info()
```


### Cabin
The missing values in this completely at Random it is quiet length to explain what it means so i will link a really good article [here](https://www.theanalysisfactor.com/missing-data-mechanism/). It is safe to say that we can ignore this data


### Embarked 
The missing data in this category is very small hence we can drop it without it affecting our final results


With that our data cleaning process is completed


## Buliding models and Testing accuracy

Here we will build different classification models and test their accuracies. This is done using the R programming language. The following models have been built:

1)Random Forest
2) Bayes-Navier Classification 
3) Decision Tree
4) Support Vector Machines
5) Generalized Logistic Regression
6) Artifical Neural Network

If you want to know more about these you can find it [here](https://techvidvan.com/tutorials/classification-in-r/)

Initalize we use a 70-30 splitting for our cleaned data. The codes grows quite big but all follow similar steps so I will put the main steps here

1) Build the model using the training dataset
2) Predict the test cases using the built model
3) Make the confusion matrix with the predicted values and the actual value
4) Calculate the accuracy = (sum(diag(confusion matrix))/ total number of test cases) * 100

Finally we plot a graph showing the accuracy levels of the different models


## Prediction
Here we finally load the test cases and try and predict whether the person survived or not. We need to correct the file as mentioned before applying our model. Once our model is applied we can save the data frame as separate csv file for final upload.