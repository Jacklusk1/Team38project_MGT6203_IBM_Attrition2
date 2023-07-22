#Clear variables and set seed to 1
rm(list = ls())
set.seed(1)

library(dplyr)
library(tidyverse)
library(plyr)
library(ggplot2)
library(glmnet)
library(kknn)
library(lme4)
library(plyr)
library(tidyr)
library(tibble)
library(tree)
library(randomForest)
library(e1071) #SVM
library(caret) #SVM
library(rpart) #Decision Tree
library(rpart.plot)

# Load the data set
data <- read.csv('IBM.csv',header=TRUE)

#Check for NA data and do some investigative analysis
apply(is.na(data), 2, sum) #No NA data points
names(data)
glimpse(data)
summary(data)

#exploratory analaysis
sum(data$Attrition == "Yes") #237 workers left the company
sum(data$Attrition == "No") #1233 workers did not leave the company

attrition_rate <- sum(data$Attrition == "Yes") / (nrow(data))
attrition_rate #16.1% of the workers left the company


## Data Manipulation ##

#First change certain binary predictor variables to 0 and 1 instead of "yes" and "no"
#data$Attrition <- ifelse(data$Attrition == "Yes", 1 , 0)
#data$OverTime <- ifelse(data$OverTime == "Yes", 1, 0)

#Remove the columns that are not important
data$EmployeeCount <- NULL
data$EmployeeNumber<- NULL
data$StandardHours <- NULL
data$Over18 <- NULL

#All the categorical variables need to be changed to factors
data$Education <- factor(data$Education)
data$EnvironmentSatisfaction <- factor(data$EnvironmentSatisfaction)
data$JobInvolvement <- factor(data$JobInvolvement)
data$JobLevel <- factor(data$JobLevel)
data$JobSatisfaction <- factor(data$JobSatisfaction)
data$PerformanceRating <- factor(data$PerformanceRating)
data$RelationshipSatisfaction <- factor(data$RelationshipSatisfaction)
data$StockOptionLevel <- factor(data$StockOptionLevel)
data$WorkLifeBalance <- factor(data$WorkLifeBalance)

#Attrition and Overtime variables need to be changed to factors
data$Attrition <- factor(data$Attrition, levels = c("No", "Yes"))
data$OverTime <- factor(data$OverTime, levels = c("No", "Yes"))

#create dataset for employees that left
subset_left <- data %>% filter(Attrition == "Yes")
summary(subset_left)

#create dataset for employees that stayed
subset_stayed <- data %>% filter(Attrition == "No")
summary(subset_stayed) 

#Create training data subset with 80% of the data
indexes <- sample(1:nrow(data), size=0.8*nrow(data)) #1176 data points
train_data <- data[indexes,]
test_data <- data[-indexes,] #294 data points

## Exploratory Data Analysis

#Let's ask the question "Do workers that complete overtime have a higher level of attrition?"
summary(data %>%
  filter(Attrition == "Yes", OverTime == "Yes"))

count(subset_left$OverTime == "Yes") / nrow(subset_left)
count(subset_stayed$OverTime == "Yes") / nrow(subset_stayed)

#Out of the employees that stayed 23.4% worked overtime and out of the employees that 
#left 53.6% employees worked overtime. There seems to be a high correlation between overtime and attrition

# Let's ask the question, "how does monthly income affect attrition"
ggplot(data, aes(x = MonthlyIncome, fill = Attrition)) + 
  geom_density(alpha = 0.5) +
  labs(x = "Monthly Income", y = "") +
  ggtitle("Attrition by income level") +
  theme_classic()


# Let's ask the question, "how does job satisfaction affect attrition"
ggplot(data, aes(x = JobSatisfaction, fill = Attrition)) + 
  geom_density(alpha = 0.5) +
  labs(x = "Job Satisfaction", y = "") +
  ggtitle("Attrition by Job Satisfaction") +
  theme_classic()



## Support Vector Machine ##


svm <- svm(train_data$Attrition ~ ., data=train_data
                 ,type="C-classification" 
                 ,kernel="radial")

#svm <- svm(train_data$Attrition ~ ., data = train_data, kernel = "radial", cost = 10, scale = FALSE)
print(svm)

#SVM hyperparameter tuning
#tuned <- tune(svm,factor(Attrition)~.,data = train_data)

#predict training data 
predict_svm <- predict(svm,newdata=test_data)

#Confusion Matrix
confusionMatrix(predict_svm,test_data$Attrition)

#The model accuracy is 84.0%

## Classification Decision Tree ##
dtree <- rpart(Attrition~., data = train_data, method = "class")

dtree

#The decision tree has main splits at TotalWorkingYears >= 1.5, OverTime= "No" , WorkLifeBalance  >=2, JobSatisfaction >=2.

#adjust margins
par(mar = c(1, 1, 1, 1))

#plot decision tree
rpart.plot(dtree)

predict_dtree <- predict(dtree,test_data,type = "class")

confusionMatrix(predict_dtree,test_data$Attrition)

#The decision tree has 78.9% model accuracy 


### Logistic Regression Model ###

#logistic models want the response variable to be either 1 or 0. 
train_data$Attrition <- ifelse(train_data$Attrition == "Yes", 1 , 0)

#linear model, even though we want to go with a logistic model
#lm_model <- lm(Attrition ~ + OverTime , data = train_data)
#summary(lm_model)

glm_model <- glm(Attrition~ Age + YearsInCurrentRole + OverTime + DistanceFromHome+Education+EnvironmentSatisfaction+ HourlyRate+JobInvolvement
                 + JobLevel+JobSatisfaction + MonthlyIncome + NumCompaniesWorked + PercentSalaryHike+ PerformanceRating +
                  TotalWorkingYears + TrainingTimesLastYear + WorkLifeBalance +YearsAtCompany + YearsSinceLastPromotion + YearsWithCurrManager, data = train_data, family = "binomial")
summary(glm_model)

#AIC = 813.47. The variables with the most significance is OverTime, DistanceFromHome,
#EnvironmentSatisfaction, JobInvolvement, JobSatisfaction, NumCompaniesWorked, WorkLifeBalance,
#YearsSinceLastPromotion.

#More importantly PercentSalaryHike, TrainingTimesLastYear,YearsAtCompany,JobLevel,
#HourlyRate, Education are not statistically significant



