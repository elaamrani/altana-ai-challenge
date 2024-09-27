# Use Case Solution Overview

This solution leverages Docker to create a consistent environment with all necessary dependencies. It features two main services:

- **Lab**: A JupyterLab container for modeling.
- **API**: A FastAPI container for serving machine learning models.

Both these services interact with an Arango DB database.

## Getting Started

To set up the application:

1. Clone the GitHub repository.
2. Run the following command to build the Docker images:

```bash
docker compose up --build
```

Note: This may take up to 10 minutes while downloading PyTorch dependencies.

## Accessing the Services

- **Jupyter Lab**: [http://127.0.0.1:8889/lab?token=lab1](http://127.0.0.1:8889/lab?token=lab1)
- **FastAPI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) (for API documentation)

## Jupyter Lab Components

1. **discovery.ipynb**: Understanding the different data models.
2. **descriptive.ipynb**: Running the network statistics.
3. **link_prediction.ipynb**: Building a basic link prediction model.

## Schema

The lab creates collections in an Arango Database:

- **Organization**: Represents sender/receiver IDs and name.
- **Country**: Contains country IDs.
- **Site**: Combines organization and country.
  - This collection uses a ML service classifying countries based on address due to missing country data.
  - The ML service calculates BERT embeddings for addresses using HuggingFace and predicts the country using PyTorch.

## Data

- Subset of raw/processed data locally stored for faster computation.

## Artifacts

- ML model artifacts for country classification and link prediction.
- You may need to run "link_prediction.ipynb" and/or "sites.ipynb" notebooks in order to save the generated ".joblib" artifacts locally for API testing

## Graph Metrics

Compute graph metrics such as degree centrality, closeness, betweenness, and page rank.

## Shutting Down Docker Services

At the end of reviewing, run the following command to stop the containers:

```bash
docker compose down
```
