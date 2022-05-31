import json
import os
import string
from collections import defaultdict
from enum import Enum
from typing import Tuple, Optional, List, Dict
from uuid import uuid4

import pandas as pd
import sklearn
import umap
from datasets import Dataset
import torch
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
import numpy as np
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

CPU = 'cpu'
CUDA = 'cuda'
PYTORCH_TENSORS = 'pt'
INPUT_IDS_FEATURE = "input_ids"
ATTENTION_MASK_FEATURE = "attention_mask"
REMOVE_PUNCTS_DICT = {ord(c): ' ' for c in string.punctuation}


class EmbeddingType(str, Enum):
    TRANSFORMER = 'transformer'
    TFIDF = 'tfidf'
    SENTENCE_TRANSFORMER = 'sentence_transformer'
    COMBINED = "combined"


class ColumnsNames(BaseModel):
    title: str = "title"
    hidden_state: str = "hidden"
    embedding: str = "embedding"
    low_dim_embedding: str = "ld_embedding"
    cluster_id: str = "cluster_id"
    cluster_label: str = "cluster_label"
    content: str = "content"
    url: str = "url"


# Config #

class ClusteringConfig(BaseModel):
    optimize_num_means: bool = True
    n_means: int = 8
    n_init: int = 10
    max_sample_size_for_optimization: int = 10000


class DimReductionConfig(BaseModel):
    n_neighbors: int = 20
    n_components: int = 5
    metric: str = "cosine"


class ExperimentConfig(BaseModel):
    data_path: str = "test_data.csv"
    seed: int = 1337
    uid: str = Field(default_factory=lambda: str(uuid4()))
    should_subsample: bool = False
    subsample_size: int = 100
    results_dir: str = "./results"
    columns_names: ColumnsNames = Field(default_factory=ColumnsNames)
    device: str = CUDA if torch.cuda.is_available() else CPU

    embedding_type: EmbeddingType = EmbeddingType.SENTENCE_TRANSFORMER.value
    transformer_inference_batch_size: int = 100
    transformer_model_checkpoint: str = "sentence-transformers/distiluse-base-multilingual-cased-v2"
    top_k_tfidf: int = 7

    num_clusters_range: Tuple[int, int] = (10, 100)
    dim_reduction: DimReductionConfig = Field(default_factory=DimReductionConfig)
    clustering: ClusteringConfig = Field(default_factory=ClusteringConfig)
    visualize_clusters: bool = True
    max_tfidf_features: int = 2000
    show_elbow: bool = False

    class Config:
        use_enum_values = True


class ResultConfig(BaseModel):
    config: ExperimentConfig
    clustering_score: float
    elbow_score: Optional[float] = None

    class Config:
        use_enum_values = True


config = ExperimentConfig()


# Config ends here #


def save_result(result_name: str, config: ExperimentConfig, dataset: Dataset, clustering_score: float,
                elbow_model=None, clusters_figure=None,
                ):
    elbow_score = None
    if elbow_model is not None:
        elbow_score = float(elbow_model.elbow_score_)

    result_config = ResultConfig(config=config, clustering_score=clustering_score, elbow_score=elbow_score)
    os.makedirs(config.results_dir, exist_ok=True)

    result_dir = os.path.join(config.results_dir, f"{result_name}_{clustering_score:.2f}_{config.uid}")
    os.makedirs(result_dir, exist_ok=False)

    result_path = os.path.join(result_dir, "result.json")
    dataset.save_to_disk(result_dir)

    df_path = os.path.join(result_dir, "results_df.csv")
    dataset.to_pandas().to_csv(df_path, encoding="utf-8")
    json.dump(result_config.dict(), open(result_path, 'w'))
    if clusters_figure is not None:
        figure_path = os.path.join(result_dir, "clusters.html")
        clusters_figure.write_html(figure_path, auto_open=False)


def remove_unwanted_terms(text):
    import re
    sites = [
        "google", "facebook", "amazon", "youtube", "survey",
        "ebay", "anonymizedname", "gmail", "bing", "suche",
        "email", "yahoo", "mail", "outlook", "umfrage", "e mail"
    ]
    url_parts = ["https", "http", "www", "com", "co", "de"]

    terms = [f"\\b{term}\\b" for term in sites + url_parts]

    res_text = re.sub(f"({'|'.join(terms)})", "", text)

    return res_text


def _load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8", index_col=0)
    df = df.dropna(subset=[config.columns_names.title, config.columns_names.url])

    clean_titles = df[config.columns_names.title].apply(canonize_text)
    clean_urls = df[config.columns_names.url].apply(canonize_text)
    empty_content_mask = clean_urls.replace('', np.nan,).isna() | \
                         clean_titles.replace('', np.nan).isna()
    df[config.columns_names.content] = clean_titles + " @ " + clean_urls
    df = df[~empty_content_mask]

    return df


def batched_tokenize(tokenizer, text_col: str):
    def apply(batch):
        res = tokenizer(batch[text_col], truncation=True, return_tensors="pt", padding=True)
        return {k: v.detach().cpu().numpy() for k, v in res.items()}

    return apply


def batched_inference(model):
    def apply(batch):
        with torch.no_grad():
            outputs = model(torch.Tensor(batch[INPUT_IDS_FEATURE]).int().to(config.device),
                            attention_mask=torch.Tensor(batch[ATTENTION_MASK_FEATURE]).to(config.device))
        res = {config.columns_names.hidden_state: outputs.last_hidden_state.detach().cpu().numpy()}
        del outputs
        return res

    return apply


def batched_dim_reduction(model, src_embd_col: str, target_embd_col: str):
    def apply(batch):
        ld_embeddings = model.transform(batch[src_embd_col])
        return {target_embd_col: ld_embeddings}

    return apply


def sample_dataset(dataset: Dataset, sample_size: int, seed: int):
    dataset = dataset.shuffle(seed=seed)
    dataset = dataset.select(range(sample_size))
    return dataset


def batched_clustering(model, embedding_col: str):
    def apply(batch):
        clusters = model.predict(batch[embedding_col])
        return {config.columns_names.cluster_id: clusters}

    return apply


def batched_mean_pooling(batch):
    hiddens = torch.Tensor(batch[config.columns_names.hidden_state])
    attention_mask = torch.Tensor(batch[ATTENTION_MASK_FEATURE])
    attention_mask_expanded = attention_mask.unsqueeze(-1).expand(hiddens.size()).float()
    pooled_hiddens = torch.sum(hiddens * attention_mask_expanded, 1) / torch.clamp(attention_mask_expanded.sum(1),
                                                                                   min=1e-9)
    return {config.columns_names.embedding: pooled_hiddens.numpy()}


def compute_cluster_labels(dataset, cluster_col, content_col, texts_to_label_func):
    cluster_id_to_texts = defaultdict(list)
    for sample in dataset:
        cluster_id_to_texts[sample[cluster_col]].append(sample[content_col])
    res = {cluster_id: texts_to_label_func(texts) for cluster_id, texts in cluster_id_to_texts.items()}
    return res


def canonize_text(text):
    text = text.lower()
    text = text.translate(REMOVE_PUNCTS_DICT)
    text = remove_unwanted_terms(text)
    text = ' '.join(text.strip().split())
    return text


def clustered_texts_to_label(texts: List[str], k=3):
    tfidf = TfidfVectorizer(max_features=config.max_tfidf_features)
    tfidf_vectors = tfidf.fit_transform(texts)
    top_k_terms = list(np.take(tfidf.get_feature_names(),
                               list(np.array(np.argsort(np.sum(tfidf_vectors, axis=0))[:, -k:][:, ::-1])[0])))
    return '-'.join(list(top_k_terms))


def generate_clusters_plot(vectors, labels, contents):
    df = pd.DataFrame.from_records(vectors)
    label_col = 'label'
    df[label_col] = labels
    n_dim = len(vectors[0])
    print(df.iloc[:5, :])
    if n_dim == 3:
        fig = px.scatter_3d(df, x=0, y=1, z=2, color=label_col, text=contents)
    elif n_dim == 2:
        fig = px.scatter(df, x=0, y=1, color=label_col, text=contents)
    else:
        raise ValueError("Plotting dimensionality must be 2 or 3")
    fig.update_traces(hoverinfo='all', selector=dict(type='scatter3d'))
    return fig


def add_transformer_embedding(dataset: Dataset, text_col: str) -> Dataset:
    model = AutoModel.from_pretrained(config.transformer_model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(config.transformer_model_checkpoint)

    dataset = dataset.map(batched_tokenize(tokenizer, text_col), batched=True)

    model.to(config.device)
    dataset = dataset.map(batched_inference(model), batched=True, batch_size=config.transformer_inference_batch_size)
    model.to(CPU)

    dataset = dataset.map(batched_mean_pooling, batched=True)
    return dataset


def train_tfidf(texts):
    tfidf_model = TfidfVectorizer(max_features=config.max_tfidf_features)
    tfidf_model.fit(texts)
    return tfidf_model


def add_tfidf_embedding(dataset: Dataset, text_col: str, **cache) -> Tuple[Dataset, Dict]:
    tfidf_model_key = "tfidf_model"
    if tfidf_model_key not in cache:
        tfidf_model = train_tfidf(dataset[text_col])
        cache[tfidf_model_key] = tfidf_model
    else:
        tfidf_model = cache[tfidf_model_key]
    tfidf_vectors = tfidf_model.transform(dataset[text_col]).toarray()
    dataset = dataset.add_column(name=config.columns_names.embedding, column=list(tfidf_vectors))
    return dataset, cache


def add_sentence_transformer_embedding(dataset: Dataset, text_col: str) -> Tuple[Dataset, Dict]:
    embedder = SentenceTransformer(config.transformer_model_checkpoint)
    dataset = dataset.map(
        lambda batch: {config.columns_names.embedding: embedder.encode(batch[text_col])}
    )
    return dataset, {}


def add_combined_embedding(dataset: Dataset, text_col: str) -> Tuple[Dataset, Dict]:
    tfidf_model = train_tfidf(dataset[text_col])
    tfidf_vectors = tfidf_model.transform(dataset[text_col]).toarray()
    sorted_tfidf_vectors = tfidf_vectors.argsort()[:, config.top_k_tfidf:][:, ::-1]
    tfids_top_k = list(
        pd.DataFrame(np.take(tfidf_model.get_feature_names(), sorted_tfidf_vectors)).apply(' , '.join, axis=1))

    embedder = SentenceTransformer(config.transformer_model_checkpoint)
    embeddings = embedder.encode(tfids_top_k, batch_size=config.transformer_inference_batch_size,
                                 show_progress_bar=True)
    dataset = dataset.add_column(config.columns_names.embedding, list(embeddings))
    return dataset, {}


embedding_registry = {
    EmbeddingType.TRANSFORMER: add_transformer_embedding,
    EmbeddingType.TFIDF: add_tfidf_embedding,
    EmbeddingType.SENTENCE_TRANSFORMER: add_sentence_transformer_embedding,
    EmbeddingType.COMBINED: add_combined_embedding
}


def run_pipe(data_path, subsample_size, should_subsample, should_optimize_num_means, visualize_clusters, save_results,
             result_name, embedding_cache=None, scaler=None,
             dim_reduction_model=None, clustering_model=None, cluster_id_to_label=None):
    # Load data
    df = _load_data(data_path)
    dataset = Dataset.from_pandas(df)

    # Subsample
    dataset = sample_dataset(dataset,
                             sample_size=subsample_size if should_subsample else len(dataset),
                             seed=config.seed)

    # Create embeddings
    if embedding_cache is None:
        embedding_cache = {}
    dataset, embedding_cache = embedding_registry[config.embedding_type](dataset, config.columns_names.content,
                                                                         **embedding_cache)

    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(dataset[config.columns_names.embedding])

    dataset = dataset.map(lambda batch: {config.columns_names.embedding:
                                             scaler.transform(batch[config.columns_names.embedding])},
                          batched=True
                          )

    # Dim reduction
    if dim_reduction_model is None:
        dim_reduction_model = umap.UMAP(n_neighbors=config.dim_reduction.n_neighbors,
                                        n_components=config.dim_reduction.n_components,
                                        metric=config.dim_reduction.metric)
        dim_reduction_model.fit(dataset[config.columns_names.embedding])

    dataset = dataset.map(batched_dim_reduction(dim_reduction_model, config.columns_names.embedding,
                                                config.columns_names.low_dim_embedding), batched=True)
    X = dataset[config.columns_names.low_dim_embedding]

    # Find best num clusters
    if should_optimize_num_means:
        clustering_optimization_sample_size = config.clustering.max_sample_size_for_optimization
        X_opt = X[:clustering_optimization_sample_size]
        X_opt = np.array(X_opt)

        optimizaton_clustering_model = KMeans()
        visualizer = KElbowVisualizer(optimizaton_clustering_model, k=config.num_clusters_range)

        visualizer.fit(X_opt)
        if config.show_elbow:
            visualizer.show()

        best_num_means = visualizer.elbow_value_
        if best_num_means is None:
            print("Didn't reach an elbow, taking max provided in range")
            best_num_means = config.num_clusters_range[1]
        config.clustering.n_means = int(best_num_means)

    # Clustering
    if clustering_model is None:
        clustering_model = KMeans(n_clusters=config.clustering.n_means, n_init=config.clustering.n_init)
        clustering_model.fit(X)

    dataset = dataset.map(batched_clustering(clustering_model, config.columns_names.low_dim_embedding),
                          batched=True)

    if cluster_id_to_label is None:
        cluster_id_to_label = compute_cluster_labels(dataset,
                                                     cluster_col=config.columns_names.cluster_id,
                                                     content_col=config.columns_names.content,
                                                     texts_to_label_func=clustered_texts_to_label)

    dataset = dataset.map(lambda sample: {config.columns_names.cluster_label:
                                              cluster_id_to_label[sample[config.columns_names.cluster_id]]})

    # Visualize clusters
    clusters_figure = None
    if visualize_clusters:
        if config.dim_reduction.n_components > 3:
            print("Cannot visualize dimension larger than 3")
        else:
            vectors = dataset[config.columns_names.low_dim_embedding]
            labels = dataset[config.columns_names.cluster_label]
            contents = dataset[config.columns_names.content]
            clusters_figure = generate_clusters_plot(vectors, labels, contents)

    clustering_score = sklearn.metrics.silhouette_score(dataset[config.columns_names.low_dim_embedding],
                                                        labels=dataset[config.columns_names.cluster_id])

    # Save results
    if save_results:
        save_result(result_name=result_name,
                    config=config,
                    dataset=dataset.remove_columns([c for c in dataset.column_names if c not in
                                                    [config.columns_names.cluster_label,
                                                     config.columns_names.cluster_id,
                                                     config.columns_names.content,
                                                     config.columns_names.title,
                                                     config.columns_names.url]]),
                    clusters_figure=clusters_figure,
                    clustering_score=clustering_score)

    return embedding_cache, scaler, clustering_model, dim_reduction_model, cluster_id_to_label


def main():
    run_pipe(data_path=config.data_path,
             subsample_size=config.subsample_size,
             should_subsample=config.should_subsample,
             should_optimize_num_means=config.clustering.optimize_num_means,
             visualize_clusters=config.visualize_clusters,
             result_name="all",
             save_results=True)


if __name__ == "__main__":
    main()
