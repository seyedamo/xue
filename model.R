library(caret)
library(data.table)

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

set.seed(42)

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

## generate features
## log based
str(cv.log.train)

cv.log.train[, date := as.Date(time, format = "%Y-%m-%d")]

save(cv.log.train, file = "./cv.log.train.RData")

names(cv.log.train)

cv.log.train[, list(enrollment_id, username, course_id), by = enrollment_id]

cv.log.train[, `:=`(duration = as.numeric(max(date) - min(date), units = "days")
                    , num.access = length(unique(date))
                    , video.count = sum(as.integer(event == 'video'))
                    , problem.count = sum(as.integer(event == 'problem'))
                    , dicussion.count = sum(as.integer(event == 'discussion'))
                    , wiki.count = sum(as.integer(event == 'wiki'))
                    )
             , by = enrollment_id]


cv.log.train[, freq := duration / num.access]

head(cv.log.train)

## course features
cv.log.train[, list(sum(as.integer(outcome == 0))
                    , sum(as.integer(outcome == 1))
                    , .N
                    ), by = course_id][, list(V1/N, V2/N), by = course_id][order(V1)]
