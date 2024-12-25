# Information_Retrieval_Course_Final_Project

Persian Document Processing and TF-IDF Search Engine

This project processes a Persian news dataset (IR_data_news_5k 2.json) by normalizing, stemming, removing stopwords, and calculating TF-IDF weights. It includes a positional index and document retrieval based on cosine similarity, optimized for Persian text processing with libraries like Hazm and Parsivar.

Prerequisites

Python 3.x

Libraries:

hazm

parsivar

numpy

scikit-learn

scipy

json

Installation

Clone the repository.

Install required libraries using pip:

pip install hazm parsivar numpy scikit-learn scipy

Ensure the dataset (IR_data_news_5k 2.json) is available in the root directory.

Usage

Run the script to preprocess the dataset:

python script_name.py

Query the processed data:
Replace "اقدامات اخیر دولت سیزدهم" with your query in the function call:

retrieve_documents("Your Query Here", k=Number of Results)

Example:

retrieve_documents("اقدامات اخیر دولت سیزدهم", 40)

Processing Overview

Normalization: Persian text is normalized using hazm.Normalizer.

Email Handling: Emails are extracted and reintegrated after tokenization.

Tokenization: Content is tokenized into words using hazm.word_tokenize.

Stopword Removal: Persian stopwords are removed.

Stemming: Words are stemmed using Parsivar.FindStems.

TF-IDF Calculation: Weights are calculated for each term using document and total frequencies.

Positional Indexing: Index created for term positions in documents.

Cosine Similarity: Queries are ranked based on cosine similarity with document vectors.

Outputs

Preprocessed dataset saved in-memory as processed_documents.

Positional index stored in positional_index.

TF-IDF weights available in tf_idf.

Search results printed with document title, URL, and similarity score.

Additional Notes

The query processing pipeline applies the same normalization, tokenization, and stemming as the document processing step to ensure consistency.

Adjust the number of results (k) as needed for the search output.

License

This project is licensed under the MIT License. See the LICENSE file for details.

