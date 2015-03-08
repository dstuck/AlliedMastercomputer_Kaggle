#Set working Directory
setwd("~/projects/titanic")
#Import the csv data
train <- read.csv("~/projects/titanic/train.csv")
test <- read.csv("~/projects/titanic/test.csv")

#Best solution everyone died!
test$Survived <- rep(0,length(test$PassengerId))

#Check the model's prediction for the training set
train$Predicted <- rep(0,length(train$PassengerId))
train$Correct[test$Survived] <- 0
train$Correct[train$Survived == train$Predicted] <-1
mean(train$Correct)

#Check the model on the test set
submit <- data.frame(PassengerID = test$PassengerId, Survived = test$Survived
write.csv(submit, file = "curSubmission.csv", row.names = FALSE)

#Second Guess Women Live Men Die
#Check the result of the model on the training set
prop.table(table(train$Sex, train$Survived),1)
train$Predicted <- 0
train$Predicted[train$Sex == 'female'] <- 1
train$Correct <- 0
train$Correct[train$Survived == train$Predicted] <-1
mean(train$Correct)


test$Survived <- 0
test$Survived[test$Sex == 'female'] <- 1
submit <- data.frame(PassengerID = test$PassengerId, Survived = test$Survived)
write.csv(submit, file = "curSubmission.csv", row.names = FALSE)

#Women and Children (age < 16)
#Check the training set
train$Under16 <- 0 
train$Under16[train$Age < 16] <- 1
train$Predicted <- 0
train$Predicted[train$Age < 16] <- 1
train$Predicted[train$Sex == 'female'] <- 1
train$Correct <- 0
train$Correct[train$Survived == train$Predicted] <-1
mean(train$Correct)
sum(train$Correct)

#Lets look into fares and Pclass
#Some binning on fares
train$binned_fare <- '30+'
train$binned_fare[train$Fare > 20 & train$Fare < 30]<-'20-30'
train$binned_fare[train$Fare > 10 & train$Fare < 20]<-'10-20'
train$binned_fare[train$Fare > 0  & train$Fare < 10]<-'0-10'
aggregate(Survived ~  binned_fare + Pclass + Sex, data=train, FUN=function(x) {sum(x)/length(x)})

#You could come up with a model here

#Lets use a decision tree!
library(rpart)
fit <- rpart(Survived ~ Pclass + Sex + Age +SibSp + Parch + Fare+ Embarked, data=train, method='class')
plot(fit)
text(fit)
install.packages('rattle')
install.packages('rpart.plot')
install.packages('RColorBrewer')
library(rattle)
library(rpart.plot)
library(RColorBrewer)
fancyRpartPlot(fit)
#Using this tree to predict
#first the training set
train$Predicted <- predict(fit, train,type='class')
train$Correct <- 0
train$Correct[train$Survived == train$Predicted] <-1
mean(train$Correct)
sum(train$Correct)
#Now the test set
Prediction <- predict(fit, test, type = "class")
submit <- data.frame(PassengerId = test$PassengerId, Survived = Prediction)
write.csv(submit, file = "myfirstdtree.csv", row.names = FALSE)

#Now showing how you can overfit
badfit <- rpart(Survived ~ Pclass + Sex + Age +SibSp + Parch + Fare+ Embarked, data=train, method='class', control=rpart.control(minsplit=2,cp=0))
fancyRpartPlot(badfit)
#Using this tree to predict
#first the training set
train$Predicted <- predict(badfit, train,type='class')
train$Correct <- 0
train$Correct[train$Survived == train$Predicted] <-1
mean(train$Correct)
sum(train$Correct)
