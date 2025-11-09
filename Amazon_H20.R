#  Amazon_H20.AI

library(agua)
library(bonsai)
library(lightgbm)
library(ranger)
library(glmnet)
library(themis)
library(glmnet)
library(tidyverse)
library(tidymodels) 
library(vroom) 
library(patchwork)
library(ggplot2) 
library(recipes) 
library(embed) 

Sys.setenv(JAVA_HOME = "C:\\PROGRA~1\\COMMON~1\\Oracle\\Java\\javapath\\java.exe")  
Sys.setenv(PATH = paste(Sys.getenv("JAVA_HOME"), "bin", sep = "/"))


library(h2o)
h2o.init()


train_data <- vroom("train.csv") 
test_data <- vroom("test.csv")

# Feature Engineering 

train_data <- train_data %>% mutate(ACTION = factor(ACTION)) 
train_data$ACTION <- relevel(train_data$ACTION, ref = "1") 

test_data <- test_data %>% mutate(across(where(is.numeric), as.factor))
# Create recipe 

my_recipe <- recipe(ACTION ~ ., data = train_data) %>%
  step_other(all_nominal_predictors(), threshold = 0.001, other = "other") %>%
  step_lencode_glm(all_nominal_predictors(), outcome = vars(ACTION))

h2o::h2o.init()

auto_model <- auto_ml() %>%
  set_engine("h2o", max_runtime_secs=180, max_models=6) %>%
  set_mode("classification")


automl_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(auto_model) %>%
  fit(data=train_data)


final_predictions <- predict(automl_wf, new_data = test_data) %>%
  bind_cols(test_data %>%
              select(id)) %>%
  rename(Action = .pred_1) %>% select(id, Action)

# Export processed dataset 

vroom_write(x = final_predictions, file = "./amazon_rf_target_1000_trees_c.csv", delim = ",")
