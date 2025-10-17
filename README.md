# Databricks SQL Assistant using PySpark, MLflow, and LLMs

This project builds an **AI-powered SQL Assistant** that understands natural language questions and automatically generates and executes SQL queries using **Databricks**, **PySpark**, and **Large Language Models (LLMs)**.

---

## 🚀 Features

- Uses **PySpark** to read table schemas from Databricks (Hive Metastore)
- Constructs a **system prompt** describing all table structures
- Integrates **Databricks LLaMA 4 Maverick** model for SQL generation
- Executes the generated SQL query using **Databricks SQL REST API**
- Logs the model using **MLflow** for versioning and deployment
- Returns structured results as JSON output

---

## 🧠 Tech Stack

- **PySpark** – for distributed data processing and schema extraction  
- **Databricks** – for cloud data management and model deployment  
- **MLflow** – for model tracking and logging  
- **LLM (LLaMA 4 Maverick)** – for natural language to SQL translation  
- **OpenAI SDK** – to interact with the Databricks LLM endpoint  
- **Python 3.10+**

---

## 📦 Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/<your-username>/databricks-sql-assistant.git
   cd databricks-sql-assistant
