from pyspark.sql import SparkSession
import os
import re
import mlflow
import mlflow.pyfunc
warehouseid = "147e835ee9c3308d"
try:
    from openai import OpenAI
except ImportError:
    os.system("pip install openai")
    from openai import OpenAI

# Initialize Spark
spark = SparkSession.builder.getOrCreate()

# Define table names and schemas
tables = {
    "customers": "hive_metastore.default.customers",
    "online_sales": "hive_metastore.default.online_sales",
    "products": "hive_metastore.default.products"
}

def get_table_schema(table_path):
    df = spark.table(table_path)
    return ", ".join([f"{field.name} ({field.dataType.simpleString()})" for field in df.schema.fields])

schemas = {name: get_table_schema(path) for name, path in tables.items()}

# Construct the system prompt
system_prompt = "You are an AI SQL assistant. Only respond to questions related to the following tables.\n\n"
for name in tables:
    system_prompt += f"Table `{name}` schema: {schemas[name]}\n"

system_prompt += (
    "There are three tables in the Catalog hive metastore/default and the tables are customers, online_sales and products. "
    "Read the structure of the tables and whatever query im asking just sql query for it dont give anything apart from it and any questions apart from these tables just simply answer out of my scope. "
    "Just print sql query in one line dont add the name sql and dont return the output with `` exlcue it."
)

# Get Databricks token
DATABRICKS_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

# ✅ Define the dynamic SQL + SQL Execution model
class SQLAssistantModel(mlflow.pyfunc.PythonModel):
    def __init__(self, token, system_prompt):
        self.token = token
        self.system_prompt = system_prompt
        self.sql_api_url = "https://adb-2856773811930092.12.azuredatabricks.net/api/2.0/sql/statements/"
        self.sql_warehouse_id = warehouseid  # 🔁 Replace this with your actual warehouse ID

    def load_context(self, context):
        self.client = OpenAI(
            api_key=self.token,
            base_url="https://adb-2856773811930092.12.azuredatabricks.net/serving-endpoints"
        )

    def execute_sql(self, query):
        import requests
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        payload = {
            "statement": query,
            "warehouse_id": self.sql_warehouse_id,
            "wait_timeout": "30s"
        }
        response = requests.post(self.sql_api_url, headers=headers, json=payload)
        if response.status_code != 200:
            raise Exception(f"SQL execution failed: {response.status_code}\n{response.text}")
        return response.json()

    def predict(self, context, model_input):
        import pandas as pd

        if isinstance(model_input, dict):
            model_input = pd.DataFrame.from_dict(model_input)

        results = []

        for question in model_input.iloc[:, 0]:
            try:
                response = self.client.chat.completions.create(
                    model="databricks-llama-4-maverick",
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": question}
                    ]
                )
                predicted_sql = response.choices[0].message.content.strip()

                if predicted_sql.lower() == "out of my scope":
                    results.append({
                        "status": "out_of_scope",
                        "query": None,
                        "result": None
                    })
                else:
                    execution_result = self.execute_sql(predicted_sql)
                    rows=execution_result.get("result",{}).get("data_array",[])
                    results.append({
                        "status": "Query and Result will be shown",
                        "query": predicted_sql,
                        "result": rows
                    })

            except Exception as e:
                results.append({
                    "status": "Error",
                    "query": None,
                    "result": str(e)
                })

        return results

# ✅ MLflow Logging (same as before)
mlflow.set_experiment("/Users/2730707@tcsteg.onmicrosoft.com/Exper")

with mlflow.start_run():
    mlflow.set_tag("task", "sql_generation_with_execution")

    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=SQLAssistantModel(DATABRICKS_TOKEN, system_prompt)
    )

print("✅ Model logged successfully. You can now register it and deploy it to a Databricks serving endpoint.")
