# Databricks notebook source
workspace_url = 'https://adb-1951422908521494.14.azuredatabricks.net/'
pat_token = dbutils.secrets.get(scope="prod_env", key="pat_token_pod")
target_git = 'https://github.com/frenchlam/DAIWT2021.git'
target_job_name = 'Prod_Modeling'

# COMMAND ----------

# DBTITLE 1,Get Repo ID of in target env
import requests

repo_id = ''
workspace_url = workspace_url.strip("/")
head = {"Authorization": f"Bearer {pat_token}"}

try :
  repo_request = requests.get(
      f"{workspace_url}/api/2.0/repos",
      headers=head,
  )
  repo_request.raise_for_status()
except requests.exceptions.HTTPError as errh:
    print ("Http Error:",errh)
except requests.exceptions.ConnectionError as errc:
    print ("Error Connecting:",errc)
except requests.exceptions.Timeout as errt:
    print ("Timeout Error:",errt)
except requests.exceptions.RequestException as err:
    print ("Something Else",err)

all_repos = repo_request.json()['repos']
for repo in all_repos : 
  if repo['url'] == target_git : repo_id = repo['id']

# COMMAND ----------

# DBTITLE 1,Get required branch 
workspace_url = workspace_url.strip("/")
head = {"Authorization": f"Bearer {pat_token}"}
payload = {
    "branch": "master"
}

try : 
  update_repo_request = requests.patch(
      f"{workspace_url}/api/2.0/repos/{repo_id}",
      json = payload,
      headers=head,
  )
  update_repo_request.raise_for_status()
except requests.exceptions.HTTPError as errh:
    print ("Http Error:",errh)
    dbutils.notebook.exit("Http Error:",errh)
except requests.exceptions.ConnectionError as errc:
    print ("Error Connecting:",errc)
    dbutils.notebook.exit("Error Connecting:",errc)
except requests.exceptions.Timeout as errt:
    print ("Timeout Error:",errt)
    dbutils.notebook.exit("Timeout Error:",errt)
except requests.exceptions.RequestException as err:
    print ("Something Else",err)
    dbutils.notebook.exit("Timeout Error:",errt)


# COMMAND ----------

# MAGIC %md 
# MAGIC ## Call Databricks Job for retrain and test

# COMMAND ----------

# DBTITLE 1,Get Job ID
workspace_url = workspace_url.strip("/")
head = {"Authorization": f"Bearer {pat_token}"}
payload = {
    "limit": "25",
}

job_id = ''

try : 
  job_list_request = requests.get(
      f"{workspace_url}/api/2.1/jobs/list",
      json = payload,
      headers=head,
  )
  update_repo_request.raise_for_status()
except requests.exceptions.HTTPError as errh:
    print ("Http Error:",errh)
    dbutils.notebook.exit("Http Error:",errh)
except requests.exceptions.ConnectionError as errc:
    print ("Error Connecting:",errc)
    dbutils.notebook.exit("Error Connecting:",errc)
except requests.exceptions.Timeout as errt:
    print ("Timeout Error:",errt)
    dbutils.notebook.exit("Timeout Error:",errt)
except requests.exceptions.RequestException as err:
    print ("Something Else",err)
    dbutils.notebook.exit("Timeout Error:",errt)
    
all_jobs = job_list_request.json()['jobs']
for job in all_jobs : 
  if job['settings']['name'] == target_job_name : job_id = job['job_id']


# COMMAND ----------

# DBTITLE 1,run_job 
workspace_url = workspace_url.strip("/")
head = {"Authorization": f"Bearer {pat_token}"}
payload = {
    "job_id": f"{job_id}",
}

try : 
  job_run_request = requests.post(
      f"{workspace_url}/api/2.1/jobs/run-now",
      json = payload,
      headers=head,
  )
  update_repo_request.raise_for_status()
except requests.exceptions.HTTPError as errh:
    print ("Http Error:",errh)
    dbutils.notebook.exit("Http Error:",errh)
except requests.exceptions.ConnectionError as errc:
    print ("Error Connecting:",errc)
    dbutils.notebook.exit("Error Connecting:",errc)
except requests.exceptions.Timeout as errt:
    print ("Timeout Error:",errt)
    dbutils.notebook.exit("Timeout Error:",errt)
except requests.exceptions.RequestException as err:
    print ("Something Else",err)
    dbutils.notebook.exit("Timeout Error:",errt)

# COMMAND ----------

dbutils.secrets.get(scope="prod_env", key="pat_token_pod")

# COMMAND ----------


