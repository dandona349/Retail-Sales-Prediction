library(dplyr)
library(ggplot2)
library(caret)
library(brnn)
library(xgboost)
library(Cubist)
library(h2o)


# Section 1 : Reading the Data --------------------------------------------

trainfilepath = "/media/abhishek/AA56-2BBB/Projects/Retail Sales Prediction/Train_UWu5bXk.csv"
testfilepath = "/media/abhishek/AA56-2BBB/Projects/Retail Sales Prediction/Test_u94Q5KV.csv"

traindata = read.csv(file = trainfilepath,
                     header = TRUE,
                     na.strings = c("","NULL","NA","na"))

testdata = read.csv(file = testfilepath,
                    header = TRUE,
                    na.strings = c("","NULL","NA","na"))

testdata$Item_Outlet_Sales = NA

combineddata = rbind(traindata,testdata)

# Section 2: Data Cleaning ------------------------------------------------

## Removing white space from column names ----------------------------------

names(combineddata) = gsub(pattern = " ",
                           replacement = "",
                           x = names(combineddata))

## Correcting data labels in Item Fat Content ------------------------------

combineddata$Item_Fat_Content =
  gsub(pattern = c("LF"),
       replacement = "LowFat",x = combineddata$Item_Fat_Content)

combineddata$Item_Fat_Content = 
  gsub(pattern = c("low fat"),
       replacement = "LowFat",x = combineddata$Item_Fat_Content)

combineddata$Item_Fat_Content = 
  gsub(pattern = c("Low Fat"),
       replacement = "LowFat",x = combineddata$Item_Fat_Content)

combineddata$Item_Fat_Content = 
  gsub(pattern = c("reg"),
       replacement = "Regular",x = combineddata$Item_Fat_Content)

combineddata$Item_Fat_Content = 
  gsub(pattern = c("regular"),
       replacement = "Regular",x = combineddata$Item_Fat_Content)

combineddata$Item_Fat_Content = as.factor(combineddata$Item_Fat_Content)

## Imputing Missing Values in Item Weight by Mean grouped by Item Identifier --------

impute.mean = function(x){replace(x,
                                  is.na(x),
                                  mean(x,na.rm = TRUE))}

combineddata = combineddata %>%
  group_by(Item_Identifier) %>%
  mutate(Item_Weight = impute.mean(Item_Weight)) %>%
  collect()

## Missing value imputation in Outlet Size ---------------------------------

combineddata$Outlet_Size[combineddata$Outlet_Identifier=="OUT045"] = "Small"
combineddata$Outlet_Size[combineddata$Outlet_Identifier=="OUT017"] = "Medium"
combineddata$Outlet_Size[combineddata$Outlet_Identifier=="OUT010"] = "Small"

## Replacing 0 in Item_Visibility by mean grouped by Item ------------------

nrow(combineddata[combineddata$Item_Visibility==0,])

impute.zero = function(x){replace(x,
                                  x==0,
                                  mean(x,na.rm = TRUE))}

combineddata =  combineddata %>%
  group_by(Item_Identifier) %>%
  mutate(Item_Visibility = impute.zero(Item_Visibility))

# Section 3: Feature Engineering -----------------------------------------------------

## Breaking Item_identifier to categories ----------------------------------

y1 = substr(x = combineddata$Item_Identifier,
            start = 1,
            stop = 2)

combineddata$Item_ID_ = as.factor(y1)

## Change fat content to Non Edible where Item id id NC --------------------

combineddata$Item_Fat_Content = as.character(combineddata$Item_Fat_Content)
combineddata$Item_Fat_Content[combineddata$Item_ID_ == "NC"] = "NonEdible"
combineddata$Item_Fat_Content = as.factor(combineddata$Item_Fat_Content)

## Change dairy in FD to dairy food ----------------------------------------

combineddata$Item_Type = as.character(combineddata$Item_Type)
combineddata$Item_Type[combineddata$Item_Type== "Dairy" &
                         combineddata$Item_ID_== "FD"] = "Dairy Food"
combineddata$Item_Type = as.factor(combineddata$Item_Type)

## Outlet years ------------------------------------------------------------

combineddata$Outlet_Years = 2013 - combineddata$Outlet_Establishment_Year
combineddata$Outlet_Years = as.factor(combineddata$Outlet_Years)
combineddata$Outlet_Establishment_Year = as.factor(combineddata$Outlet_Establishment_Year)

## Mrp diff by Item ---------------------------------------------------------------

avg_mrp = combineddata %>%
  group_by(Item_Identifier) %>%
  summarise(avg_mrp = mean(Item_MRP))

avg_mrp.match = match(combineddata$Item_Identifier,
                      avg_mrp$Item_Identifier)

combineddata$Item_MRP_Dev = combineddata$Item_MRP -
  avg_mrp$avg_mrp[avg_mrp.match]

## Creating Sub categories of Item Identifier ------------------------------

y2 = substr(x = combineddata$Item_Identifier,
            start = 3,
            stop = 3)

combineddata$Item_ID_Subcat = as.factor(y2)

## Creating Sub-Sub categories of Item_ID ----------------------------------

y3 = substr(x = combineddata$Item_Identifier,
            start = 4,
            stop = 5)

combineddata$Item_ID_SubSubcat = as.factor(y3)

## Normalizing Item Weight -------------------------------------------------

combineddata$Item_Weight = scale(x = combineddata$Item_Weight,
                                 center = TRUE)[,1]


## Normalizing Item Visibility ---------------------------------------------

combineddata$Item_Visibility = scale(x = combineddata$Item_Visibility,
                                     center = TRUE)[,1]

## Normalizing Item MRP ----------------------------------------------------

combineddata$Item_MRP = scale(x = combineddata$Item_MRP,
                              center = TRUE)[,1]

## Create dummy variables --------------------------------------------------

### Item type ---------------------------------------------------------------

combineddata = as.data.frame(combineddata)

combineddata = cbind(combineddata,
                     model.matrix(~Item_Type-1,
                                  data = combineddata))

### Item_Fat_Content --------------------------------------------------------

combineddata = cbind(combineddata,
                     model.matrix(~Item_Fat_Content-1,
                                  data = combineddata))

### Outlet Identifier -------------------------------------------------------

combineddata = cbind(combineddata,
                     model.matrix(~Outlet_Identifier-1,
                                  data = combineddata))

### Outlet location type ----------------------------------------------------

combineddata = cbind(combineddata,
                     model.matrix(~Outlet_Location_Type-1,
                                  data = combineddata))

### Outlet Type -------------------------------------------------------------

combineddata = cbind(combineddata,
                     model.matrix(~Outlet_Type-1,
                                  data = combineddata))

### Item ID -----------------------------------------------------------------

combineddata = cbind(combineddata,
                     model.matrix(~Item_ID_-1,
                                  data = combineddata))

### Item_ID_Subcat ----------------------------------------------------------

combineddata = cbind(combineddata,
                     model.matrix(~Item_ID_Subcat-1,
                                  data = combineddata))

### Outlet_Years ------------------------------------------------------------

combineddata = cbind(combineddata,
                     model.matrix(~Outlet_Years-1,
                                  data = combineddata))

### Item_ID_SubSubcat -------------------------------------------------------

combineddata = cbind(combineddata,
                     model.matrix(~Item_ID_SubSubcat-1,
                                  data = combineddata))

# Section 4: Feature Selection -------------------------------------------------------

names(combineddata) = gsub(pattern = " ",
                           replacement = "",
                           x = names(combineddata))

h2o.init(nthreads = -1,min_mem_size = "5g")

trainhex = as.h2o(combineddata[1:8523,-c(1,3,5,7,8,9,10,11,13,14,16,17)])

cat("start rf estimation\n")

features <- h2o.colnames(trainhex)

## Partition the training set
trainhex.split <- h2o.splitFrame(trainhex, ratios=.8)

## Grid Search for Model Comparison
ntrees_opt <- c(50,100,500,1000)
maxdepth_opt <- c(5,10,20,50)
hyper_parameters <- list(ntrees=ntrees_opt,
                         max_depth=maxdepth_opt)

if(!file.exists("/media/abhishek/AA56-2BBB/Projects/Retail Sales Prediction/Objects/FeatureSelection.rds")){
  grid1 <- h2o.grid("randomForest", 
                    hyper_params = hyper_parameters,
                    y = "Item_Outlet_Sales",
                    x = features,
                    seed = 123,
                    training_frame = trainhex.split[[1]],
                    validation_frame = trainhex.split[[2]])
  
}else{
  grid1 = readRDS(file = "/media/abhishek/AA56-2BBB/Projects/Retail Sales Prediction/Objects/FeatureSelection.rds")
}

saveRDS(object = grid1,file = "/media/abhishek/AA56-2BBB/Projects/Retail Sales Prediction/Objects/FeatureSelection.rds")

## print out all prediction errors and run times of the models
grid1

## print out the rmse for all of the models
model_ids <- grid1@model_ids
rmse <- vector(mode="numeric", length=0)
grid_models <- lapply(model_ids, function(model_id) { model = h2o.getModel(model_id) })
for (i in 1:length(grid_models)) {
  print(sprintf("rmse: %f", h2o.rmse(grid_models[[i]])))
  rmse[i] <- h2o.rmse(grid_models[[i]])
}

best_id <- unlist(model_ids[order(rmse,decreasing=F)][1])

fit.best <- h2o.getModel(model_id = best_id)

selected = h2o.varimp(fit.best)[1:25,1]

h2o.varimp_plot(fit.best,num_of_features = 25)

# Section 5: Models ----------------------------------------

## Create training and test data -------------------------------------------

combineddata = as.data.frame(combineddata)

selected = gsub(pattern = " ",
                replacement = "",
                x = selected)

names(combineddata) = gsub(pattern = " ",
                           replacement = "",
                           x = names(combineddata))

modeltraindata = combineddata[1:8523,c(selected,"Item_Outlet_Sales")]

modeltestdata = combineddata[8524:14204,selected]

## Model1 : Simple Linear Regression Model ---------------------------------
# RMSE : 1483.214

model1 = lm(Item_Outlet_Sales ~ Item_MRP,
            data = modeltraindata)

model1pred = predict(object = model1,
                     newdata = modeltestdata)


submission = data.frame(Item_Identifier = testdata$Item_Identifier,
                        Outlet_Identifier = testdata$Outlet_Identifier)

submission$Item_Outlet_Sales = abs(model1pred)

write.csv(x = submission,file = "/media/abhishek/AA56-2BBB/Projects/Retail Sales Prediction/Submissions2/submission1-SLR-2.csv",row.names = FALSE)

## Model 2: Multinomial Linear Regression ----------------------------------
# RMSE: 1191.534

model2 = lm(Item_Outlet_Sales~Item_MRP + 
              Outlet_TypeGroceryStore + 
              Outlet_TypeSupermarketType3 + 
              Outlet_TypeSupermarketType1 + 
              Item_ID_SubSubcat55 + 
              Item_ID_SubcatY,
            data = modeltraindata)

model2pred = predict(object = model2,
                     newdata = modeltestdata)

submission = data.frame(Item_Identifier = testdata$Item_Identifier,
                        Outlet_Identifier = testdata$Outlet_Identifier)

submission$Item_Outlet_Sales = abs(model2pred)

write.csv(x = submission,file = "/media/abhishek/AA56-2BBB/Projects/Retail Sales Prediction/Submissions2/submission2-MLR-NT.csv",row.names = FALSE)

# Model 2 Cross Validation ------------------------------------------------
# RMSE  :1191.534

slrtr = trainControl(method = "cv",number = 10,verboseIter = TRUE)

model2tune = train(x = modeltraindata[,c("Item_MRP",
                                         "Outlet_TypeGroceryStore",
                                         "Outlet_TypeSupermarketType3",
                                         "Outlet_TypeSupermarketType1",
                                         "Item_ID_SubSubcat55",
                                         "Item_ID_SubcatY")],
                   y = modeltraindata$Item_Outlet_Sales,
                   method = "lm",
                   metric = "RMSE",
                   maximize = FALSE,
                   trControl = slrtr)

model2tunnedpred = abs(predict(object = model2tune,
                           newdata = modeltestdata))

submission = data.frame(Item_Identifier = testdata$Item_Identifier,
                        Outlet_Identifier = testdata$Outlet_Identifier)

submission$Item_Outlet_Sales = model2tunnedpred

write.csv(x = submission,file = "/media/abhishek/AA56-2BBB/Projects/Retail Sales Prediction/Submissions2/submission2-MLR-T.csv",row.names = FALSE)

## Model 3 : Xgboost -------------------------------------------------------
# RMSE: 1188.582

model3 = xgboost(data = data.matrix(modeltraindata[,-26]),
                 label = modeltraindata$Item_Outlet_Sales,
                 nrounds = 2000,
                 early_stopping_rounds = 5,
                 params = list(booster = "gblinear",
                               lambda = 0.5,
                               lambda_bias = 0.02,
                               alpha = 0.01,
                               objective = "reg:linear",
                               eval_metric = "rmse"),
                 verbose = 2)

model3pred = predict(object = model3,
                     newdata = data.matrix(modeltestdata),
                     ntreelimit = 0)

submission = data.frame(Item_Identifier = testdata$Item_Identifier,Outlet_Identifier = testdata$Outlet_Identifier)

submission$Item_Outlet_Sales = abs(model3pred)

write.csv(x = submission,file = "/media/abhishek/AA56-2BBB/Projects/Retail Sales Prediction/Submissions2/submission3-XGB-NT.csv",row.names = FALSE)

## Model 4: Xgboost Tuned --------------------------------------------------
# RMSE : 1218.671

model4trcontrol = trainControl(method = "cv",
                       number = 10,
                       allowParallel = TRUE,
                       verboseIter = TRUE)

xggrid = expand.grid(nrounds = c(100,500,1000),
                     lambda = c(0.001,0.005,0.01,0.1,1),
                     alpha = c(0,0.0001,0.001,0.01,0.1,1),
                     eta = c(0.001,0.01,0.1))
                    
model4 = train(Item_Outlet_Sales~.,
              data = modeltraindata,
              method = "xgbLinear",
              tuneGrid = xggrid,
              trControl = model4trcontrol)

model4pred = predict(object = model4,newdata = data.matrix(modeltestdata),ntreelimit = 0)

submission = data.frame(Item_Identifier = testdata$Item_Identifier,Outlet_Identifier = testdata$Outlet_Identifier)

submission$Item_Outlet_Sales = abs(model4pred)

write.csv(x = submission,file = "/media/abhishek/AA56-2BBB/Projects/Retail Sales Prediction/Submissions2/submission3-XGB-T.csv",row.names = FALSE)


## Model 5: Cubist -----------------------------------------------------------------
# RMSE : 1153.3581

model5 = cubist(x = modeltraindata[,-26],
                y = modeltraindata$Item_Outlet_Sales,
                committees = 1,
                neighbors = 5)

model5pred = predict(object = model5,
                     newdata = modeltestdata)

submission = data.frame(Item_Identifier = testdata$Item_Identifier,Outlet_Identifier = testdata$Outlet_Identifier)

submission$Item_Outlet_Sales = abs(model5pred)

write.csv(x = submission,file = "/media/abhishek/AA56-2BBB/Projects/Retail Sales Prediction/Submissions2/submission5-Cubist-NT-2.csv",row.names = FALSE)


## Model 6: Cubist Tuned ---------------------------------------------------
# RMSE: 1152.750
model6trcontrol = trainControl(method = "cv",
                               number = 10,
                               allowParallel = TRUE,
                               verboseIter = TRUE)
model6tune = expand.grid(committees = c(1,5,50,100)
                           ,neighbors = c(0,1,3,7,9))
model6 = train(x = modeltraindata[,-26],
                           y = modeltraindata$Item_Outlet_Sales,
                           method = "cubist",
                           metric = "rmse",
                           maximize = FALSE,
                           tuneGrid = model6tune,
                           trControl = model6trcontrol)

model6pred = predict(object = model6,
                     newdata = modeltestdata)

submission = data.frame(Item_Identifier = testdata$Item_Identifier,Outlet_Identifier = testdata$Outlet_Identifier)

submission$Item_Outlet_Sales = abs(model6pred)

write.csv(x = submission,file = "/media/abhishek/AA56-2BBB/Projects/Retail Sales Prediction/Submissions2/submission6-Cubist-T.csv",row.names = FALSE)

## Model 7: Bayesian Regularized Neural Network ----------------------------
# RMSE: 1137.294

model7 = brnn(Item_Outlet_Sales~.,
              data = modeltraindata,
              neurons = 8,
              epochs = 5000,
              mu = 0.005,
              min_grad = 1e-10,
              cores = 4)

model7pred = predict(object = model7,
                     newdata = modeltestdata)

submission = data.frame(Item_Identifier = testdata$Item_Identifier,Outlet_Identifier = testdata$Outlet_Identifier)

submission$Item_Outlet_Sales = abs(model7pred)

write.csv(x = submission,file = "/media/abhishek/AA56-2BBB/Projects/Retail Sales Prediction/Submissions2/submission7-BRNN-NT.csv",row.names = FALSE)

## Model 8: BRNN Tuned -----------------------------------------------------
# RMSE : 1146.895

model8trcontrol = trainControl(method = "cv",
                        number = 10,
                        allowParallel = TRUE)

model8 = train(Item_Outlet_Sales~.,
                   data = modeltraindata,
                   method = "brnn",
                   tuneGrid = expand.grid(neurons = c(5,6,7,8,9,10)),
                   trControl = model8trcontrol)

model8pred = predict(object = model8,
                     newdata = modeltestdata)

submission = data.frame(Item_Identifier = testdata$Item_Identifier,Outlet_Identifier = testdata$Outlet_Identifier)

submission$Item_Outlet_Sales = abs(model8pred)

write.csv(x = submission,file = "/media/abhishek/AA56-2BBB/Projects/Retail Sales Prediction/Submissions2/submission8-BRNN-T.csv",row.names = FALSE)
