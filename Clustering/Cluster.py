### Clustering Module : Based on KMeans, Agglomerative Clustering
### Model : snunlp/KR-SBERT-V40K-klueNLI-augSTS

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

