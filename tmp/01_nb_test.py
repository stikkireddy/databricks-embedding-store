# Databricks notebook source
# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

# MAGIC %pip install langchain sentence_transformers faiss-gpu

# COMMAND ----------

# MAGIC %sql
# MAGIC SET spark.sql.adaptive.coalescePartitions.enabled=false

# COMMAND ----------

from delta_embeddings.store import DeltaLakeEmbeddingStore, DefaultEmbeddingsFaissIndex
dls = DeltaLakeEmbeddingStore(table_name="hive_metastore.sri_demo_catalog.embeddings_v6", spark_session=spark)

# COMMAND ----------

# first column is id as string and second item is content as string
input_data = spark.sql("SELECT review_id, review_body FROM hive_metastore.sri_demo_catalog.amazon_luggage_reviews_cleaned ORDER BY review_id DESC limit 100;")
dls.add_and_embed_documents(input_data=input_data, sentence_transformer_model_name="sentence-transformers/all-MiniLM-L12-v2")

# COMMAND ----------

dls.self_sim_search("sentence-transformers/all-MiniLM-L12-v2", num_partitions=6).display()

# COMMAND ----------


