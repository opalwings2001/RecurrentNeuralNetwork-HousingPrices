#Recurrent Neural Network to Calculate Housing Prices in Kings County, WA

library(keras)
library(tfdatasets)
library(ggplot2)
library(tidyverse)

#set random seed for reproduction
#remove column with un-scaled prices and randomly sample the data into two sets of identical 
#size
set.seed(1)
housing_kings_2<-housing_2[,-4]
housing_kings_2
smp<-sample(1:nrow(housing_kings_2),nrow(housing_kings_2)/2)
ncol(housing_kings_2)

#split into training data and testing data, the target being the prices column
train.data<-housing_kings_2[smp,-3]
train.data
train.target<-housing_kings_2[smp,3]
train.target
test.data<-housing_kings_2[-smp,-3]
test.data
test.target<-housing_kings_2[-smp,3]
test.target

dim(train.data)
summary(train.target)

#normalize the attributes, since they each have a different scale
spec <- feature_spec(train.data, label ~ . ) %>% 
  step_numeric_column(all_numeric(), normalizer_fn = scaler_standard()) %>% 
  fit()

spec

#examine output of dense_features layer to see scaled values in a Tensor matrix
layer <- layer_dense_features(
  feature_columns = dense_features(spec), 
  dtype = tf$float32
)
layer(train.data)

#create model, input as a list of Keras input layers from training data
input <- layer_input_from_dataset(train.data)
input

#output includes hidden layers and output layers with specified dropout levels
output <- input %>% 
  layer_dense_features(dense_features(spec)) %>% 
  layer_dense(units = 32, activation = "relu") %>%
  layer_dropout(0.20) %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dropout(0.20) %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dropout(0.10) %>%
  layer_dense(units = 1) 

#create model
model2 <- keras_model(input, output) 

  
summary(model2)

#compile model
model2 %>% 
  compile(
    loss = "mse",
    optimizer = optimizer_rmsprop(),
    metrics = list("mae")
  )

model2

#instantiate a variable to stop the training of the model when validation loss does not improve
#this will prevent overfitting
early_stop <- callback_early_stopping(monitor = "val_loss", patience = 20)

#train the model with fit and specify number of epochs, validation split and batch size
trained_model<-model2 %>% fit(
  x = train.data,
  y = train.target,
  batch_size = 512,
  epochs = 400,
  validation_split = 0.2,
  callbacks = list(early_stop))

#plot with ggplot to see trendlines
plot(trained_model)

#evaluate on test data
model2 %>%
  evaluate(test.data,test.target)

#generate predicted values and write in a csv
test_pred<-model2 %>% predict(test.data)
head(test_pred)
test_pred

preds<-cbind(test.data$date,test_pred)
preds

write.csv(preds,"predicted_prices.csv")


#For time series, easier to have on entry per date. Excel manipulation was done to get just the unique dates
p<-predicted_prices
#removed duplicated dates using duplicated() from tidyverse
p_clean<-p[!duplicated(p$date),]
#make p_clean a timeseries
pricets<-ts(p_clean)
pricets
#plot prices
plot(pricets[,3],main="Predicted Housing Prices from 2014 to 2015")

housing_data<-housing_kings_2[,-3]
head(housing_data)
housing_target<-housing_kings_2[,3]
head(housing_target)
predictions<-model2 %>% predict(housing_kings_2[,-3])
head(predictions)

plot(predictions,housing_target,xlab = "Predicted Prices (Ten-Thousands)",ylab = "Actual Prices (Ten-Thousands)",main="Actual vs. Predicted Housing Prices")
abline(a=0,b=1)
