from pathlib import Path

import numpy as np
import pytest

from tutorial.wham_dswp import (
    WHAM_EMBEDDING_DIM,
    cluster_embeddings,
    load_annotation_labels,
    reduce_embeddings,
    validate_embeddings,
)


def test_validate_embeddings_accepts_wham_shape():
    embeddings = np.zeros((4, 3, WHAM_EMBEDDING_DIM), dtype=np.float32)
    validate_embeddings(embeddings)


def test_validate_embeddings_rejects_wrong_dimension():
    embeddings = np.zeros((4, 3, 16), dtype=np.float32)
    with pytest.raises(ValueError, match="Expected WhAM dimension"):
        validate_embeddings(embeddings)


def test_pca_and_kmeans_are_reproducible():
    rng = np.random.default_rng(42)
    embeddings = rng.normal(size=(12, WHAM_EMBEDDING_DIM)).astype(np.float32)
    first_projection = reduce_embeddings(embeddings, method="pca")
    second_projection = reduce_embeddings(embeddings, method="pca")
    assert np.allclose(first_projection, second_projection)

    result = cluster_embeddings(embeddings, n_clusters=3)
    assert result["labels"].shape == (12,)
    assert -1.0 <= result["silhouette"] <= 1.0


def test_annotation_join_requires_explicit_columns(tmp_path: Path):
    annotation_path = tmp_path / "annotations.csv"
    annotation_path.write_text("file,Unit\n1.wav,unit-a\n", encoding="utf-8")
    paths, labels = load_annotation_labels(annotation_path, [tmp_path / "1.wav"])
    assert paths == [tmp_path / "1.wav"]
    assert labels.tolist() == ["unit-a"]
