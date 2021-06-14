# Loading the ML models

SVM <- readRDS("Support Vector.rds")
forest <- readRDS("random forest.rds")

# Predicting
test.data$Survived = round(predict(forest,test.data))

# Extracting only the columns we need and saving it
final.forest <- cbind.data.frame(test.data$PassengerId, test.data$Survived)
colnames(final.forest)<- c("PassengerId", "Survived") 
write.csv(final.forest,"E:/Projects/Titanic/data/final-forest.csv",row.names= FALSE)