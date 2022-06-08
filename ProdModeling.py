# Databricks notebook source
# MAGIC %md
# MAGIC # DAIS 2021 Data Science session: Modeling
# MAGIC 
# MAGIC Auto ML generated a baseline model for us, but, we could already see it was too simplistic. From that working modeling code, the data scientist could iterate and improve it by hand.
# MAGIC 
# MAGIC ** ... time passes ... **

# COMMAND ----------

model_name = "dais-2021-churn_MLA"
experiment_name = "/Users/matthieu.lamairesse@databricks.com/experiments/DAIWT2021/"

# COMMAND ----------

# from mlflow.tracking.client import MlflowClient

# all_experiments = [exp.experiment_id for exp in MlflowClient().list_experiments()]

# for experiment in all_experiments : 
#   experiment = MlflowClient().get_experiment(experiment)
#   if experiment.name == experiment_name : experiment_id = experiment.experiment_id

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Calculate our features

# COMMAND ----------

# Using our custom function 

from functions import compute_service_features

demographics_df = spark.read.table("matthieulamDAIWT.demographic")
enriched_df = compute_service_features(spark.read.table("matthieulamDAIWT.other_features"))
training_set = demographics_df.join(enriched_df, demographics_df.customerID == enriched_df.customerID, "inner").drop(enriched_df.customerID)

df_loaded = training_set.drop("customerID").toPandas()
display(df_loaded)

# COMMAND ----------

# # Using Feature Store 
# from databricks.feature_store import FeatureStoreClient, FeatureLookup

# fs = FeatureStoreClient()

# training_set = fs.create_training_set(spark.read.table("matthieulamDAIWT.demographic"), 
#                                       [FeatureLookup(table_name = "matthieulamDAIWT.service_features", lookup_key="customerID")], 
#                                       label="Churn", exclude_columns="customerID")
# df_loaded = training_set.load_df().toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC This is the same as the code produced by auto ML, to define the model:

# COMMAND ----------

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

def build_model(params):
  transformers = []

  bool_pipeline = Pipeline(steps=[
      ("cast_type", FunctionTransformer(lambda df: df.astype(object))),
      ("imputer", SimpleImputer(missing_values=None, strategy="most_frequent")),
      ("onehot", OneHotEncoder(handle_unknown="ignore")),
  ])
  transformers.append(("boolean", bool_pipeline, 
                       ["Dependents", "PaperlessBilling", "Partner", "PhoneService", "SeniorCitizen"]))

  numerical_pipeline = Pipeline(steps=[
      ("converter", FunctionTransformer(lambda df: df.apply(pd.to_numeric, errors="coerce"))),
      ("imputer", SimpleImputer(strategy="mean"))
  ])
  transformers.append(("numerical", numerical_pipeline, 
                       ["AvgPriceIncrease", "Contract", "MonthlyCharges", "NumOptionalServices", "TotalCharges", "tenure"]))

  one_hot_pipeline = Pipeline(steps=[
      ("imputer", SimpleImputer(missing_values=None, strategy="constant", fill_value="")),
      ("onehot", OneHotEncoder(handle_unknown="ignore"))
  ])
  transformers.append(("onehot", one_hot_pipeline, 
                       ["DeviceProtection", "InternetService", "MultipleLines", "OnlineBackup", \
                        "OnlineSecurity", "PaymentMethod", "StreamingMovies", "StreamingTV", "TechSupport", "gender"]))

  xgbc_classifier = XGBClassifier(
    n_estimators=int(params['n_estimators']),
    learning_rate=params['learning_rate'],
    max_depth=int(params['max_depth']),
    min_child_weight=params['min_child_weight'],
    random_state=810302555
  )

  return Pipeline([
      ("preprocessor", ColumnTransformer(transformers, remainder="passthrough", sparse_threshold=0)),
      ("standardizer", StandardScaler()),
      ("classifier", xgbc_classifier),
  ])

# COMMAND ----------

from sklearn.model_selection import train_test_split

target_col = "Churn"
split_X = df_loaded.drop([target_col], axis=1)
split_y = df_loaded[target_col]

X_train, X_val, y_train, y_val = train_test_split(split_X, split_y, train_size=0.9, random_state=810302555, stratify=split_y)

# COMMAND ----------

# MAGIC %md
# MAGIC Here, we use `hyperopt` to perform some 'auto ML' every time the model is rebuilt, to fine tune it. This is similar to what auto ML did to arrive at the initial baseline model.

# COMMAND ----------

from hyperopt import fmin, hp, tpe, SparkTrials, STATUS_OK
import numpy as np
from sklearn.metrics import log_loss, accuracy_score
import mlflow

mlflow.autolog(log_input_examples=False)
#mlflow.set_experiment(experiment_name)

def train_model(params):
  model = build_model(params)
  model.fit(X_train, y_train)
  loss = log_loss(y_val, model.predict_proba(X_val))
  mlflow.log_metrics({'log_loss': loss, 'accuracy': accuracy_score(y_val, model.predict(X_val))})
  return { 'status': STATUS_OK, 'loss': loss }
  
search_space = {
  'max_depth':        hp.quniform('max_depth', 3, 10, 1),
  'learning_rate':    hp.loguniform('learning_rate', -5, -1),
  'min_child_weight': hp.loguniform('min_child_weight', 0, 2),
  'n_estimators':     hp.quniform('n_estimators', 50, 500, 10)
}

best_params = fmin(fn=train_model, space=search_space, algo=tpe.suggest, \
                   max_evals=64, trials=SparkTrials(parallelism=8))

# COMMAND ----------

# MAGIC %md
# MAGIC Now, build one last model on all the data, with the best hyperparams. The model is logged in a 'feature store aware' way, so that it can perform the joins at runtime. The model doesn't need to be fed the features manually.

# COMMAND ----------

import mlflow
from mlflow.models.signature import infer_signature

mlflow.autolog(log_input_examples=True)
#mlflow.set_experiment(experiment_name)

split_X = df_loaded.drop([target_col], axis=1)
split_y = df_loaded[target_col]

model = build_model(best_params)
model.fit(split_X, split_y)

with mlflow.start_run() as run:
  
  df_loaded = training_set.load_df().toPandas()
  split_X = df_loaded.drop([target_col], axis=1)
  split_y = df_loaded[target_col]

  model = build_model(best_params)
  model.fit(split_X, split_y)
  
  best_run = run.info

# COMMAND ----------

# # with Feature store 
# import mlflow
# from mlflow.models.signature import infer_signature
# from databricks.feature_store import FeatureStoreClient, FeatureLookup

# fs = FeatureStoreClient()

# mlflow.autolog(log_input_examples=True)
# #mlflow.set_experiment(experiment_name)

# with mlflow.start_run() as run:
#   training_set = fs.create_training_set(spark.read.table("matthieulamDAIWT.demographic"), 
#                                       [FeatureLookup(table_name = "matthieulamDAIWT.service_features", lookup_key="customerID")], 
#                                       label="Churn", exclude_columns="customerID")
  
#   df_loaded = training_set.load_df().toPandas()
#   split_X = df_loaded.drop([target_col], axis=1)
#   split_y = df_loaded[target_col]

#   model = build_model(best_params)
#   model.fit(split_X, split_y)

# # Log Feature store transformation information with MLFLow
#   fs.log_model(
#     model,
#     "model",
#     flavor=mlflow.sklearn,
#     training_set=training_set,
#     registered_model_name=model_name,
#     input_example=split_X[:100],
#     signature=infer_signature(split_X, split_y))
  
#   best_run = run.info

# COMMAND ----------

# MAGIC %md
# MAGIC The process above created a new version of the registered model `dais-2021-churn`. Transition it to Staging.

# COMMAND ----------

import mlflow.tracking

client = mlflow.tracking.MlflowClient()

model_version = client.get_latest_versions(model_name, stages=["None"])[0]
client.transition_model_version_stage(model_name, model_version.version, stage="Staging")

# COMMAND ----------


