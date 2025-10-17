# Logistic Regression Using Random Forests

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

rf_mod <- rand_forest(mtry = tune(),
                      min_n=tune(),
                      trees=500) %>%
set_engine("ranger") %>%
set_mode("classification")

# Create recipe
my_recipe <- recipe(ACTION ~ ., data = train_data) %>%
  # Collapse rare categories (<0.1%)
  step_other(all_nominal_predictors(), threshold = 0.001, other = "other") %>%
  # Target encoding
  step_lencode_glm(all_nominal_predictors(), outcome = vars(ACTION))



rf_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(rf_mod)

## Grid of values to tune over
tuning_grid <- grid_regular(
  mtry(range = c(1, ncol(train_data) - 1)),
  min_n(),
  levels = 3
)

## Split data for CV
folds <- vfold_cv(train_data, v = 3, repeats=1)

## Run the CV
CV_results <- rf_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics = metric_set(roc_auc, accuracy))

## Find Best Tuning Parameters
bestTune <- CV_results %>%
  select_best(metric = "roc_auc")

## Finalize the Workflow & fit it
final_wf <-
  rf_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=train_data)

## Predict
final_predictions <- final_wf %>%
  predict(new_data = test_data, type = "prob") %>%
  bind_cols(test_data %>% select(id)) %>%
  rename(Action = .pred_1) %>%   # Assuming you want P(ACTION = 1)
  select(id, Action)

# Export processed dataset
vroom_write(x = final_predictions, file = "./amazon_rf.csv", delim = ",")
