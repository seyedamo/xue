library(caret)
library(data.table)
library(randomForest)
library(ROCR)
library(doMC)

registerDoMC(cores = 2)


## training
enrol.train <- fread("../data/enrollment_train.csv")
log.train <- fread("../data/log_train.csv")
object <- fread("../data/object.csv")
truth.train <- fread("../data/truth_train.csv")

## testing
enrol.test <- fread("../data/enrollment_test.csv")
log.test <- fread("../data/log_test.csv")

## combine enrol.train with outcome flag
setnames(truth.train, c("enrollment_id", "outcome"))

setkey(enrol.train, enrollment_id)
enrol.train <- enrol.train[truth.train]

names(enrol.train)
enrol.train[, .N, by = outcome]
truth.train[, .N, by = outcome]

## benchmark
## good students' behaviour



## split to training and evaluation

set.seed(386)

inTraining <- createDataPartition(enrol.train$outcome, p = 0.8, list = FALSE)
cv.enrol.train <- enrol.train[inTraining[, 1]]
cv.enrol.test <- enrol.train[-inTraining[, 1]]

dim(cv.enrol.train)
dim(cv.enrol.test)

setkey(log.train, enrollment_id)

cv.log.train <- log.train[cv.enrol.train][, c("i.username", "i.course_id") := NULL]
cv.log.test <- log.train[cv.enrol.test][, c("i.username", "i.course_id") := NULL]

cv.log.train[, .N, by = enrollment_id]
cv.log.test[, .N, by = enrollment_id]

dim(cv.log.train)
names(cv.log.train)
names(cv.log.test)
## generate features
## log based
save(cv.log.train, file = "./cv.log.train.RData")


## explore

cv.log.train[, list(enrollment_id, username, course_id), by = enrollment_id]

## ?? proportion of videos / problems  covered

## video count is significant
cv.log.train[event == "video", list(.N), by = list(course_id, outcome)][order(course_id, outcome)]


## person's duration is very significant
cv.log.train[, list(duration = max(date) - min(date)), by = list(course_id, outcome, username)][, list(mean(duration)), by = list(course_id, outcome)][order(course_id, outcome)]

## new features
add_new_features <- function(dt) {
    dt[, date := as.Date(time, format = "%Y-%m-%d")]
    new_dt <- dt[, list(duration = as.numeric(max(date) - min(date), units = "days")
                        , num.access = length(unique(date))
                        , video.count = sum(as.integer(event == 'video'))
                        , problem.count = sum(as.integer(event == 'problem'))
                        , dicussion.count = sum(as.integer(event == 'discussion'))
                        , wiki.count = sum(as.integer(event == 'wiki'))
                        , course_id = unique(course_id)
                        )
               , by = enrollment_id][, freq.access := duration / num.access]
    new_dt[, `:=`(course_id = as.factor(course_id))]

}

training <- add_new_features(cv.log.train)
dim(training)
dim(cv.enrol.train)
setkey(training, enrollment_id)
training <- training[cv.enrol.train]
training[, `:=`(outcome = factor(outcome, levels = c(0,1), labels = c("No", "Yes")))]


testing <- add_new_features(cv.log.test)
names(testing)


## course features
cv.log.train[, list(sum(as.integer(outcome == 0))
                    , sum(as.integer(outcome == 1))
                    , .N
                    ), by = course_id][, list(V1/N, V2/N), by = course_id][order(V1)]

## some courses are harder
cv.enrol.train[, list(length(unique(username))), by = list(course_id, outcome)][order(course_id, outcome)]

## Model training
training[, username:= NULL]
training[, enrollment_id:= NULL]
training[, course_id.1:=NULL]
str(training)




fit.rf <- randomForest(outcome ~., data = training, ntree = 500)
predict.rf <- predict(fit.rf, type = "prob")

auc <- function(predict, target) {
    rocr <- prediction(predict[, 2], target)
    roc <- performance(rocr, "tpr", "fpr")
    plot(roc, colorize = TRUE)
    performance(rocr, "auc")@y.values
}

auc(predict.rf, training$outcome)

test.predict.rf <- predict(fit.rf, type = "prob", newdata = testing)

auc(test.predict.rf, cv.enrol.test$outcome)


## Cross-validation by Caret

training <- as.data.frame(training)

ctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 5, classProbs = TRUE, allowParallel = TRUE, summaryFunction = twoClassSummary, verboseIter = TRUE)

##ctrl <- trainControl(method = "cv", number = 10, classProbs = TRUE, summaryFunction = twoClassSummary, verboseIter = TRUE)
##ctrl <- trainControl(method = "none", number = 10, classProbs = TRUE, summaryFunction = twoClassSummary, verboseIter = TRUE)

## random forest

cv.rf.fit <- train(outcome ~.
                 , data = training
                 , method = 'rf'
                 , ntrees = 500
                 , trControl = ctrl
                 , metric = 'ROC'
                 , importance = TRUE)

cv.rf.fit

## grid search for parameters
grid <- expand.grid(.mtry = c(24))

cv.rf.fit <- train(outcome ~.
                 , data = training
                 , method = 'rf'
                 , ntrees = 500
                 , trControl = ctrl
                 , metric = 'ROC'
                 , importance = TRUE
                 , tuneGrid = grid)

cv.rf.fit

plot(cv.rf.fit)

test.predict.cv.rf <- predict(cv.rf.fit, type = "prob", newdata = testing)
auc(test.predict.cv.rf, cv.enrol.test$outcome)


######### Decision trees
## C5.0   package: C50

cv.c50.fit <- train(outcome ~.
                 , data = training
                 , method = 'C5.0'
                 , trControl = ctrl
                 , metric = 'ROC'
                 , importance = TRUE)

cv.rf.fit

## CART package: rpart
cv.rpart.fit <- train(outcome ~.
                 , data = training
                 , method = 'rpart'
                 , trControl = ctrl
                 , metric = 'ROC'
                 )

cv.rf.fit

## Neural Networks package:
cv.nnet.fit <- train(outcome ~.
                 , data = training
                 , method = 'nnet'
                 , trControl = ctrl
                 , metric = 'ROC'
                 )


## SVM
cv.svm.fit <- train(outcome ~.
                 , data = training
                 , method = 'svmLinear'
                 , trControl = ctrl
                 , metric = 'ROC'
                 )

cv.svm.fit

## Naive Bayes
grid <- expand.grid(.fL = c(1, 2, 3), .usekernel = c(TRUE, FALSE))

cv.nb.fit <- train(outcome ~.
                 , data = training
                 , method = 'nb'
                 , trControl = ctrl
                 , metric = 'ROC'
                   , tuneGrid = grid
                 )


cv.nb.fit

######## submission #######

## build model
## training
training <- add_new_features(log.train)
setkey(training, enrollment_id)
training <- training[enrol.train]
training[, `:=`(outcome = as.factor(outcome))]
training[, username:= NULL]
training[, enrollment_id:= NULL]
training[, course_id.1:=NULL]
str(training)

## testing

testing <- add_new_features(log.test)
dim(testing)
str(testing)

fit.rf <- randomForest(outcome ~., data = training, ntree = 100)

predict.rf <- predict(fit.rf, type = "prob", newdata = testing)

head(predict.rf)

submission <- cbind(testing[, list(enrollment_id)], predict.rf[, 2])
head(submission)
setnames(submission, c("enrollment_id", "prob"))

dim(submission)
##write.csv(submission, file = "../data/RFsub.csv", row.names = FALSE, col.names = FALSE)
write.table(submission, file = "../data/RFsub.csv", row.names = FALSE, col.names = FALSE, sep = ",")









## fun


p <- 1000 / dim(enrol.train)[1]

inTraining <- createDataPartition(enrol.train$outcome, p = p, list = FALSE)
training <- enrol.train[inTraining[, 1]]

dim(training)
