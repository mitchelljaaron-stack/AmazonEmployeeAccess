# Support Vector Machines Amazon Dataset Analysis

library(glmnet)

library(tidyverse)

library(tidymodels)

library(vroom)

library(patchwork)

library(ggplot2)

library(recipes)

library(embed)

train_data <- vroom("train.csv")

test_data <- vroom("test.csv")

# Feature Engineering

train_data <- train_data %>%
  mutate(
    ACTION = as.factor(ACTION),
    across(where(is.numeric) & !all_of("ACTION"), as.factor)
  )

test_data <- test_data %>%
  mutate(across(where(is.numeric), as.factor))

# Create recipe
my_recipe <- recipe(ACTION ~ ., data = train_data) %>%
  # Collapse rare categories (<0.1%)
  step_other(all_nominal_predictors(), threshold = 0.001, other = "other") %>%
  # Target encoding
  step_lencode_glm(all_nominal_predictors(), outcome = vars(ACTION))

## SVM models3
svmPoly <- svm_poly(degree=tune(), cost=tune()) %>% # set or tune
  set_mode("classification") %>%
set_engine("kernlab")

svmRadial <- svm_rbf(rbf_sigma=tune(), cost=tune()) %>% # set or tune
  set_mode("classification") %>%
set_engine("kernlab")

svmLinear <- svm_linear(cost=tune()) %>% # set or tune
  set_mode("classification") %>%
set_engine("kernlab")

svm_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(svmPoly)

## Grid of values to tune over
tuning_grid <- grid_regular(cost(),
                            levels = 3) ## L^2 total tuning possibilities

## Split data for CV
folds <- vfold_cv(train_data, v = 3, repeats=1)

## Run the CV
CV_results <- logReg_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics = metric_set(roc_auc, accuracy))

## Find Best Tuning Parameters
bestTune <- CV_results %>%
  select_best(metric = "roc_auc")

## Finalize the Workflow & fit it
final_wf <-
  logReg_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=train_data)

## Predict
final_predictions <- final_wf %>%
  predict(new_data = test_data, type = "prob") %>%
  bind_cols(test_data %>% select(id)) %>%
  rename(Action = .pred_1) %>%   # Assuming you want P(ACTION = 1)
  select(id, Action)

# Export processed dataset
vroom_write(x = final_predictions, file = "./amazon_pen_mix_PCA_logReg.csv", delim = ",")
