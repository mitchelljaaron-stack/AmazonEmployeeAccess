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
  step_lencode_glm(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  #Everything numeric for SMOTE so encode it here
  step_smote(all_outcomes(), neighbors=K) %>%
  step_upsample()

## SVM models


svm_rbf <- svm_rbf(rbf_sigma = 0.177, cost = 0.00316) %>%
  set_mode("classification") %>%
  set_engine("kernlab")

svmPoly <- svm_poly(degree = 1, cost = 0.0131) %>%
  set_mode("classification") %>%
  set_engine("kernlab")

svm_linear <- svm_linear(cost = 0.0131) %>%
  set_mode("classification") %>%
  set_engine("kernlab")

svm_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(svm_linear)


## Finalize the Workflow & fit it
final_wf <- 
  svm_wf %>%
  fit(data=train_data)

## Predict
final_predictions <- final_wf %>%
  predict(new_data = test_data, type = "prob") %>%
  bind_cols(test_data %>% select(id)) %>%
  rename(Action = .pred_1) %>%
  select(id, Action)

# Export processed dataset
vroom_write(x = final_predictions, file = "./amazon_svm_linear.csv", delim = ",")
