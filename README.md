# Similarity

This is a scaffold for a natural language interface with company data.

Architecture:
- Postgres with pgvector as the content repository
- FastAPI API that exposes
  - create embeddings endpoint
  - reranking endpoint
  - similarity search (vector similarity, and vector + reranker)
  - similarity search (same) on specific entities
  - chat!
- Workers for ingestion and processing of:
  - issues
  - posts
  - website
  - glossary
- batch processes that:
  - generates questions based on the documentation
  - generates keywords based on the documentation
  - summarizes the contents of the entities

## How should you run this?

1) you should first set up your own litellm proxy (ensure you have a config file in litellm/config)

```
docker run -it --rm -p 4000:4000 -v $PWD/config/litellm/config.yaml:/app/config.yaml -e OPENAI_API_KEY=<youropenaiapikey> litellm/litellm:v1.74.0-stable --config /app/config.yaml
```

the config file should be something like

```
model_list:
  - model_name: openai-nano
    litellm_params:
      model: openai/gpt-4.1-nano
      api_key: os.environ/OPENAI_API_KEY
      api_version: "1"
  - model_name: openai-mini
    litellm_params:
      model: openai/gpt-4.1-mini
      api_key: os.environ/OPENAI_API_KEY
      api_version: "1"

litellm_settings:
  cache: True
  cache_params:
    type: local

```

2) ensure you have a postgres with pgvector running

```
docker run -it -p 5432:5432 -e POSTGRES_PASSWORD=github_similarity_search -e POSTGRES_USER=github_similarity_search -e POSTGRES_DB=github-similarity-search -v ${PWD}/similarity-search-db:/var/lib/postgresql/data pgvector/pgvector:pg17
```

3) copy .env.example to .env

```
cp .env.example .env
```

4) install Astral's uv package manager https://docs.astral.sh/uv/getting-started/installation/

5) run `uv sync` on the root to pull dependencies

6) run `uv run run.py manage-db --enable-vector` to enable the vector extensions

7) run `uv run run.py manage-db --recreate`, to create the database

8) run `uv run run.py populate` to start populating the data


## Embeddings

You can run `uv run run.py embeddings all` to start creating embeddings on the data after it has been pulled

## LLM batches

Run `uv run run.py batch create issues` to create the LLM summaries for the issues table (you can then create the embeddings for these summaries as well)


## TODO

- semantic compression (we should send context through a model first to compress text and avoid duplication of semantics)
- review the entire code to see if there are abstractions that might be useless
- review the queries
- check if we're adding the indexes (don't think so)