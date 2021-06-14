require(c("caret","e1071","caTools","party","randomForest","ISLR"))
#loading the needed libraries
library(easypackages)
libraries("caret","e1071","caTools","party","randomForest","ISLR","neuralnet")

#splitting the dataset
head(data,5)
set.seed(445)

split = sample.split(data$Survived, SplitRatio = .7)

train = subset(data,split == T)
test = subset(data, split == F)


# Building the tree model
tree2 = ctree(formula = Survived ~ Age+Pclass+(Sex)+Parch+SibSp+Fare+PassengerId, data = train)
plot(tree)

pred = predict(tree2,test)
acc = addmargins( table (round(pred),test$Survived))
accuracy.tree2 = (acc[1,1] +acc[2,2])/268 *100


#Building the random forest model
forest = randomForest(formula = Survived ~ Age+Pclass+Sex+Parch+SibSp+Fare, data = train)

pred2 = predict(forest,test)
acc2 = addmargins(table(round(pred2), test$Survived))
accuracy.forest = (acc2[1,1] +acc2[2,2])/268 * 100

#Building the SVM model
SVM = svm(formula = Survived ~ Age+Pclass+as.factor(Sex)+Parch+SibSp+Fare,data = train)

pred3 = predict(SVM, test)
acc3 = addmargins(table(round(pred3), test$Survived))
accuracy.svm = (acc3[1,1] +acc3[2,2])/268 * 100 


# Building navies-bayes classifier
navies.bayes = naiveBayes(formula = Survived ~ Age+Pclass+as.factor(Sex)+Parch+SibSp+Fare, data = train)

pred4 = predict(navies.bayes, test)
acc4 = addmargins(table((pred4), test$Survived))
accuracy.navies.bayes = (acc4[1,1] +acc4[2,2])/268 * 100 

#Building generalized logistic regression
logistic.reg = glm(formula = Survived ~ Age+Pclass+as.factor(Sex)+Parch+SibSp+Fare, data = train)

pred5 = predict(logistic.reg, test)
acc5 = addmargins(table(round(pred5), test$Survived))
accuracy.logi = (acc5[1,1] +acc5[2,2])/268 * 100


# Building neural network
nn = neuralnet(formula = Survived ~Age+Parch+Pclass+Fare+SibSp,data = train,
               hidden = 3,act.fct = 'logistic',linear.output = F)


pred6 = predict(nn,test)
acc6 = addmargins(table(round(pred6),test$Survived))
accuracy.nn = (acc6[1,1] +acc6[2,2])/268 * 100



# Plotting the various model
names <- c("Random Forest",  "Logistic Regression", "Navies Bayes classification", "Neural Network", "Support Vector Machines","Decision Tree")
accuracies <- c(accuracy.forest,accuracy.logi,accuracy.navies.bayes,accuracy.nn,accuracy.svm,accuracy.tree)

accuracy.table <- data.frame(names,accuracies)

ggplot(accuracy.table,aes(x = names, y = accuracies, fill = names)) +
  geom_bar(size = 5, stat = 'identity') +
  ggtitle("Accuracy of different models")


# Saving the svm and random forest models
saveRDS(SVM, "Support Vector.rds")
saveRDS(forest,"random forest.rds")

