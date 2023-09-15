# Utilities
## Highlights

- `pHAN.py`: implementation of perspective-based hierarchical attention network

- `sentence_scorer.py`: implementation of the perspective score (called topic score in this notebook), claim score (used to calculate statement score), and evidence score.

- `bertopic.py`: custom **BERTopic** functions for BERTopic to receive the segmentation results from **Articut**. Warning: these functions might not work for higher version (>0.11.0) of BERTopic.

## Others
- `articut.py`: wrapper for calling segmentation API provided by **Articut**
- `coverage.py`: calculate the percentage of how many key tokens a piece of text contains
- `data.py`: I/O utilities for the data
- `get_path.py`: to get paths based on variables in the notebooks
- `io.py`: printing and debugging in the notebooks
- `mmr.py`: to calculate maximal marginal relevance
- `next_sentence_prediction.py`: use the BERT model to predict if one sentence is the next sentence of another sentence.
- `preprocess.py`: preprocess the text
- `read_application.py`: read the application
- `torch.py`: torch dataset classes