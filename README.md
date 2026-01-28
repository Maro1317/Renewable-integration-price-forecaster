# Renewable-integration-price-forecaster

Objective: Build a machine learning system that predicts the "Net Load" (Total Demand minus
Wind/Solar Generation) and the resulting Wholesale Electricity Price 24 hours in advance.

Technical Architecture:
• Data Engineering: Construct a pipeline using requests, and pandas to ingest the XML/CSV feeds.
• Store this in a time-series optimized database or AWS S3 bucket.
• Making use of XGBoost & SkLearn in scripting in Python.
