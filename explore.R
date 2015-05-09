library(ggplot2)
library(data.table)
library(dplyr)
library(caret)

## training
enrol.train <- fread("../data/enrollment_train.csv")
log.train <- fread("../data/log_train.csv")
object <- fread("../data/object.csv")
truth.train <- fread("../data/truth_train.csv")

## testing
enrol.test <- fread("../data/enrollment_test.csv")
log.test <- fread("../data/log_test.csv")


setnames(truth.train, c("enrollment_id", "outcome"))

dim(enrol.train)
dim(log.train)
dim(truth.train)

setkey(enrol.train, enrollment_id)
setkey(log.train, enrollment_id)

enrol.train[truth.train]
log.train[truth.train]

str(enrol.train)
str(log.train)
str(object)
str(truth.train)

summary(enrol.train)
summary(log.train)
summary(object)
summary(truth.train)

## feature generating

log.train

sample <- log.train[enrollment_id %in% c(1:5)]

str(sample)

sample[, .N, by = enrollment_id]
names(sample)

## Out
## PubDate
head(blogs$PubDate)
?strptime

weekdays()
dt$date$hour

as.Date(head(blogs$PubDate), format = "%Y-%m-%d")

head(blogs$PubDate)
a <- strptime(blogs$PubDate, format = "%Y-%m-%d %H:%M:%S")
head(a)
head(a$hour)
head(weekdays(a))

blogs[, PubDate := as.Date(PubDate, format = "%Y-%m-%d")]
test.submit[, PubDate := as.Date(PubDate, format = "%Y-%m-%d")]

## in



sample <- log.train[enrollment_id %in% c(1:5)]

sample

sample[, strptime(time, format = "%Y-%m-%dT%H:%M:%S")]

strptime(sample$time, format = "%Y-%m-%dT%H:%M:%S")

sample[, date := as.Date(time, format = "%Y-%m-%d")]
## sample[, time := strptime(time, format = "%Y-%m-%dT%H:%M:%S")]


sample[, list(min(date), max(date)), by = enrollment_id]

sample[, list(length(unique(date))), by = enrollment_id]

new <- sample[, list(duration = as.numeric(max(date) - min(date), units = "days")
                     , num.access = length(unique(date)))
              , by = enrollment_id]

new[, freq := duration / num.access]

new

sample[, list(sum(as.integer(event == 'video'))
              ,sum(as.integer(event == 'problem'))
              ,sum(as.integer(event == 'discussion'))
              ,sum(as.integer(event == 'wiki')))
       , by = enrollment_id]


unique(log.train$event)




