# DSWP to WhAM embeddings tutorial

`wham_dswp_embeddings.ipynb` is a small, reproducible starter pipeline:

1. Download selected public DSWP WAV files from Hugging Face
2. Inspect their audio metadata
3. Extract mean-pooled layer embeddings with the official VampNet interface
4. Run PCA, UMAP or t-SNE and a KMeans baseline
5. Run optional coda or social-unit classifiers only when verified annotation labels are supplied

The public DSWP release has 1,501 audio files and a single `audio` feature. Its card says that there is no per-file metadata and that the recordings come from one vocal clan. Therefore the notebook does not invent social-unit labels or coda negatives. The supervised section requires a separate CSV with `file` and `Unit` columns. Coda detection additionally requires a verified `coda_label` column with positive and negative examples.

## Run

From the repository root:

```bash
python -m pip install -r tutorial/requirements.txt
jupyter notebook tutorial/wham_dswp_embeddings.ipynb
```

For real WhAM embeddings, download the official `coarse.pth` and `codec.pth` into `vampnet/models`. The notebook keeps model extraction disabled until both files are present. These files are large, so the default first run only validates public DSWP ingestion and the pipeline plumbing.

The official model code also needs the dependencies described by the repository root `setup.py` and `vampnet/setup.py`. A CPU run is intended for a tiny smoke sample. A larger embedding sweep should use a compatible GPU environment.

The first real CPU smoke run is recorded in [VALIDATION.md](VALIDATION.md). It loaded the official coarse and codec weights, extracted a finite layer-10 vector with shape `(1, 1280)` from public DSWP audio, and executed the notebook on three files.

## Sources

- [WhAM repository](https://github.com/Project-CETI/wham)
- [WhAM paper](https://arxiv.org/abs/2512.02206)
- [Public DSWP dataset](https://huggingface.co/datasets/orrp/DSWP)
- [WhAM model weights](https://doi.org/10.5281/zenodo.17633708)
