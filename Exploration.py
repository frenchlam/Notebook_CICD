# Databricks notebook source
# MAGIC %md
# MAGIC # DAIS 2021 Data Science session: Exploration
# MAGIC 
# MAGIC Welcome to Databricks! This session will illustrate a fictional, simple, but representative day in the life of a data scientist on Databricks, who starts with data and ends up with a basic production service.
# MAGIC 
# MAGIC ## Problem: Churn
# MAGIC 
# MAGIC Imagine the case of a startup telecom company, with customers who unfortunately sometimes choose to terminate their service. It would be useful to predict when a customer might churn, to intervene. Fortunately, the company has been diligent about collecting data about customers, which might be predictive. This is new territory, the first time the company has tackled the problem. Where to start?
# MAGIC 
# MAGIC ## Data Exploration
# MAGIC 
# MAGIC We can start by simply reading the data and exploring it. There's already some useful information in the `demographic` table: customer ID, whether they have churned (or not, yet), and basic demographic information:

# COMMAND ----------

# MAGIC %md
# MAGIC Do the normal, predictable things. Compute summary stats. Plot some values. See what's what.

# COMMAND ----------

display(spark.read.table("matthieulamDAIWT.demographic").summary())

# COMMAND ----------

display(spark.read.table("matthieulamDAIWT.other_features").summary())

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

from functions import compute_service_features

demographics_df = spark.read.table("matthieulamDAIWT.demographic")
enriched_df = compute_service_features(spark.read.table("matthieulamDAIWT.other_features"))
demographic_service_df = demographics_df.join(enriched_df, demographics_df.customerID == enriched_df.customerID, "inner").drop(enriched_df.customerID)

demographic_service_df.drop("customerID").write.mode("overwrite").format("delta").saveAsTable("matthieulamDAIWT.demographic_service")

display(demographic_service_df)


# COMMAND ----------

demographic_service_df.printSchema()

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Altenatively use Feature Store

# COMMAND ----------

from databricks.feature_store import FeatureStoreClient, FeatureLookup

fs = FeatureStoreClient()
training_set = fs.create_training_set(spark.read.table("matthieulamDAIWT.demographic"), 
                                      [FeatureLookup(table_name = "matthieulamDAIWT.service_features", lookup_key="customerID")], 
                                      label=None, exclude_columns="customerID")

training_set.load_df().write.mode("overwrite").format("delta").saveAsTable("matthieulamDAIWT.demographic_service")

display(training_set.load_df())

# COMMAND ----------


