import pyspark.sql.functions as F

def compute_service_features(data):
  # Count number of optional services enabled, like streaming TV
  @F.pandas_udf('int')
  def num_optional_services(*cols):
    return sum(map(lambda s: (s == "Yes").astype('int'), cols))
  
  # Below also add AvgPriceIncrease: current monthly charges compared to historical average
  service_cols = [c for c in data.columns if c not in ["gender", "SeniorCitizen", "Partner", "Dependents", "Churn"]]
  return data.select(service_cols).fillna({"TotalCharges": 0.0}).\
    withColumn("NumOptionalServices",
        num_optional_services("OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies")).\
    withColumn("AvgPriceIncrease",
        F.when(F.col("tenure") > 0, (F.col("MonthlyCharges") - (F.col("TotalCharges") / F.col("tenure")))).otherwise(0.0))


