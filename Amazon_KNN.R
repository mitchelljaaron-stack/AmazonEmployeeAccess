# Amazon Predictions Using K Nearest Neighbors

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

# KNN Model
knn_model <- nearest_neighbor(neighbors=3) %>%
  set_mode("classification") %>%
set_engine("kknn")

# Create recipe
my_recipe <- recipe(ACTION ~ ., data = train_data) %>%
  # Collapse rare categories (<0.1%)
  step_other(all_nominal_predictors(), threshold = 0.001, other = "other") %>%
  # Target encoding
  step_lencode_glm(all_nominal_predictors(), outcome = vars(ACTION))



knn_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(knn_model)


## Finalize the Workflow & fit it
final_wf <-
  knn_wf %>%
  fit(data=train_data)

## Predict
final_predictions <- final_wf %>%
  predict(knn_wf, new_data=test_data, type="prob") %>%
  bind_cols(test_data %>% select(id)) %>%
  rename(Action = .pred_1) %>%   # Assuming you want P(ACTION = 1)
  select(id, Action)

# Export processed dataset
vroom_write(x = final_predictions, file = "./amazon_knn.csv", delim = ",")
