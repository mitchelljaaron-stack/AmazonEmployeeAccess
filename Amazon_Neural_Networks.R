# Neural Networks Model Predictions

library(tensorflow)
library(keras)
library(reticulate)
library(tidymodels)
library(vroom)
library(embed)
library(ggplot2)
library(dplyr)

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

# Define model
nn_model <- mlp(
  hidden_units = tune(),
  epochs = 50
) %>%
  set_engine("keras") %>%
  set_mode("classification")

# Recipe
my_recipe <- recipe(ACTION ~ ., data = train_data) %>%
  step_other(all_nominal_predictors(), threshold = 0.001, other = "other") %>%
  step_lencode_glm(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  step_range(all_numeric_predictors(), min = 0, max = 1)

# Workflow
nn_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(nn_model)

# Grid for tuning
nn_tuneGrid <- grid_regular(
  hidden_units(range = c(1, 10)),
  levels = 3
)

# Cross-validation
folds <- vfold_cv(train_data, v = 3)

# Tune
CV_results <- nn_wf %>%
  tune_grid(
    resamples = folds,
    grid = nn_tuneGrid,
    metrics = metric_set(roc_auc, accuracy)
  )

# Select best
bestTune <- CV_results %>%
  select_best(metric = "roc_auc")

# Finalize and fit
tuned_nn <- nn_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = train_data)

# Predict on test
final_predictions <- tuned_nn %>%
  predict(new_data = test_data, type = "prob") %>%
  bind_cols(test_data %>% select(id)) %>%
  rename(Action = .pred_1) %>%
  select(id, Action)

# Plot accuracy
CV_results %>%
  collect_metrics() %>%
  filter(.metric == "accuracy") %>%
  ggplot(aes(x = hidden_units, y = mean)) +
  geom_line() +
  geom_point()

# Export
vroom_write(final_predictions, "./amazon_nn.csv", delim = ",")