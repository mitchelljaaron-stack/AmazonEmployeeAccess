# AmazonEmployeeAccess Logistic Regression

library(tidyverse)

library(tidymodels)

library(vroom)

library(patchwork)

library(ggplot2)

library(recipes)

train_data <- vroom("C:\\Users\\mitch\\OneDrive\\Documents\\GitHub\\AmazonEmployeeAccess\\train.csv")

test_data <- vroom("C:\\Users\\mitch\\OneDrive\\Documents\\GitHub\\AmazonEmployeeAccess\\test.csv")

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
  # One-hot encode
  step_dummy(all_nominal_predictors())

logRegModel <- logistic_reg() %>%
  set_engine("glm")

logReg_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(logRegModel) %>%
  fit(data=train_data)

amazon_predictions <- predict(logReg_wf,
                              new_data=test_data,
                              type="class")

final_predictions <- amazon_predictions %>%
  rename(Action = .pred_class) %>%
  bind_cols(test_data %>% select(id))

# Export processed dataset
vroom_write(x = final_predictions, file = "./amazon_logReg.csv", delim = ",")
