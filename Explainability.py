# Databricks notebook source
# MAGIC %md
# MAGIC # Model Insights with SHAP
# MAGIC 
# MAGIC As part of the exploration process, having a baseline model early helps explore the _data_ in turn. For example, the basic SHAP plots created by auto ML can be expanded to explore more of the data:

# COMMAND ----------

# MAGIC %pip install scikit-learn==1.0

# COMMAND ----------

model_name = "dais-2021-churn_MLA"

# COMMAND ----------

import mlflow
import mlflow.sklearn
from shap import KernelExplainer, summary_plot
from sklearn.model_selection import train_test_split
import pandas as pd

mlflow.autolog(disable=True)

sample = spark.read.table("matthieulamDAIWT.demographic_service").sample(0.05).toPandas()
data = sample.drop(["Churn"], axis=1)
labels = sample["Churn"]
X_background, X_example, _, y_example = train_test_split(data, labels, train_size=0.5, random_state=42, stratify=labels)

model =  mlflow.sklearn.load_model(f"models:/{model_name}/staging")
#model = mlflow.sklearn.load_model("runs:/b53bf168667a46aaadb6a06aab0bc0ac/model")

predict = lambda x: model.predict_proba(pd.DataFrame(x, columns=X_example.columns))[:,-1]
explainer = KernelExplainer(predict, X_example)
shap_values = explainer.shap_values(X=X_example)

# COMMAND ----------

summary_plot(shap_values, features=X_example)

# COMMAND ----------

# MAGIC %md
# MAGIC ... or explore how churn factors differ by gender:

# COMMAND ----------

from shap import group_difference_plot

group_difference_plot(shap_values[y_example == 1], X_example[y_example == 1]['gender'] == 'Male', feature_names=X_example.columns)

# COMMAND ----------

# MAGIC %md
# MAGIC ... or 'cluster' customers by their model explanations and see whether patterns emerge. There are clearly a group of churners that tend to be on 1-month contracts, short tenure, and equally a cluster of non-churners who are on 2-year contracts and have been long time customers. But we knew that!

# COMMAND ----------

import seaborn as sns
from sklearn.manifold import TSNE

embedded = TSNE(n_components=2, learning_rate='auto', init='pca', random_state=42).fit_transform(shap_values)

sns.set(rc = {'figure.figsize':(16,9)})
sns.scatterplot(x=embedded[:,0], y=embedded[:,1], \
                style=X_example['Contract'], \
                hue=X_example['tenure'], \
                size=(y_example == 1), size_order=[True, False], sizes=(100,200))

# COMMAND ----------


