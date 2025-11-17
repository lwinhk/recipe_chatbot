
# Recipe Knowledge Retrieval Using Embedding-Based Augmented Generation in Conversational Agents

This is a project to implement a chat bot that answer recipe for user's query from the private dataset using embedding models such as TFIDF, all-MiniLM-L6-v2 and text-embedding-3-small, and retrieval augmented generation using large language models like gpt-4.1-mini and gpt-4o-mini.

# Installation

Install required python packages.
```
pip install sentence-transformers
pip install faiss-cpu
pip install openai
pip install pandas
pip install scikit-learn joblib
```

# Dependencies
```
import pandas as pd
import numpy as np
import uuid
import json
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import faiss, time
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
import joblib
import matplotlib.pyplot as plt
```
# Data
[3A2M Cooking Recipe Dataset](https://www.kaggle.com/datasets/nazmussakibrupol/3a2m-cooking-recipe-dataset) is used in this experiment. Dataset includes 1,312,865 recipes but due to the resource limitation, the subset of the dataset is used.\
The dataset has 5 labels: title - 
* the name of the food, 
* directions - step-by-step description of the food recipes, 
* NER - ingredients of cooking recipes, 
* genre - assigned category (representation in string format), and 
* label - numeric representation of the genres.

# Files
* files/inputs/3A2M_Text.csv = recipe dataset
* files/outputs/recipes-index-all-MiniLM-L6-v2.bin = index file of all-MiniLM-L6-v2 vectors
* files/outputs/recipes-index-text-embedding-3-small = index file of text-embedding-3-small vectors
* files/outputs/recipes-tfidf.joblib = TFIDF vector file
* files/outputs/recipes.jsonl = JSON formatted recipe data files
* embed.ipynb = python program to embed dataset to different types of vectors
* query.ipynb = python program to use LLM models to generate results for input query and to measure process time of LLM models
* evaluation.ipynb = python program to evaluate embedding models and visualize the evaluation in plots

# Running
* Place files in the correct structure
* To run embed.ipynb, evaluation.ipynb and query.ipynb, use Jupyter Notebook in favorite code editor like VSCode.

# Support
For support, contact the developer at "dpc3559@autuni.ac.nz" or "lwinhk2008@gmail.com".

# Thank You
