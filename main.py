import json
import re
from hazm import *
from parsivar import FindStems
from collections import Counter
import math
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import norm
import numpy as np

# Initialize stemmer
stemmer = FindStems()

# Initialize normalizer
normalizer = Normalizer()

# Load Persian stopwords
stopwords = stopwords_list()

# Open the file
with open('IR_data_news_12k.json', 'r') as f:
    # Load JSON data from file
    data = json.load(f)

# Prepare a list to hold the processed documents
processed_documents = []

# Iterate over each document
for doc_id, doc in data.items():
    # Get the content
    content = doc.get('content', '')

    # Normalize the content
    normalized_content = normalizer.normalize(content)

    # Remove punctuation but keep email addresses intact
    normalized_content = re.sub(r'(?<!\S)\W+|\W+(?!\S)', ' ', normalized_content)

    # Extract email addresses
    emails = re.findall(r'\S+@\S+', normalized_content)

    # Remove email addresses from the content
    normalized_content = re.sub(r'\S+@\S+', '', normalized_content)

    # Tokenize the content
    words = word_tokenize(normalized_content)

    # Add the email addresses back to the list of words
    words.extend(emails)

    # Calculate word frequencies
    word_freq = Counter(words)

    # Get the 50 most common words
    most_common_words = [word for word, freq in word_freq.most_common(50)]

    # Remove the 50 most common words
    words = [word for word in words if word not in most_common_words]

    # Remove stopwords
    words = [word for word in words if word not in stopwords]

    # Stemming
    stemmed_words = [stemmer.convert_to_stem(word) for word in words]

    # Prepare a dictionary to hold the processed document
    processed_doc = {
        'id': doc_id,
        'url': doc.get('url', ''),
        'tag': doc.get('tags', []),
        'title': doc.get('title',''),
        'category':doc.get('category' , ''),
        'content': stemmed_words
    }

    # Add the processed document to the list
    processed_documents.append(processed_doc)

    """output_processed_documents = open(' output_processed_documents' , wb)
    pickle.dumb(processed_documents ,output_processed_documents) 
   """
# Now "processed_documents" is a list of dictionaries, each containing the ID, URL, tag, and normalized, stemmed content of a document
######print(processed_documents)

# Initialize an empty dictionary to hold the positional index
positional_index = {}

# Initialize a dictionary to hold the document frequency of each word
doc_freq = {}

# Initialize a counter to hold the total frequency of each word
total_freq = Counter()

# Iterate over each processed document
for doc in processed_documents:
    # Get the ID and content of the document
    doc_id = doc['id']
    content = doc['content']

    # Iterate over each word in the content
    for position, word in enumerate(content):
        # If the word is not in the index yet, add it with an empty dictionary
        if word not in positional_index:
            positional_index[word] = {}

        # If the document is not in the word's list yet, add it with an empty list
        if doc_id not in positional_index[word]:
            positional_index[word][doc_id] = []

        # Add the position to the word's list for this document
        positional_index[word][doc_id].append(position)

        # Increase the frequency of this word in this document
        if word in doc_freq:
            if doc_id in doc_freq[word]:
                doc_freq[word][doc_id] += 1
            else:
                doc_freq[word][doc_id] = 1
        else:
            doc_freq[word] = {doc_id: 1}

        # Increase the total frequency of this word
        total_freq[word] += 1

# Now "positional_index" is a dictionary where each key is a word, and the value is another dictionary.
# In this inner dictionary, each key is a document ID, and the value is a list of positions where the word appears in the document.
#print("Positional Index:")
#print(positional_index)

# Now "doc_freq" is a dictionary where each key is a document ID, and the value is a Counter.
# In this Counter, each key is a word, and the value is the frequency of the word in the document.
#print("Document Frequencies:")
#print(doc_freq)

# Now "total_freq" is a Counter where each key is a word, and the value is the total frequency of the word in all documents.
#print("Total Frequencies:")
#print(total_freq)


# Initialize an empty dictionary to hold the tf-idf weights
tf_idf = {}

# Calculate the number of documents
N = len(processed_documents)

# Iterate over each word in the document frequencies
for word, docs in doc_freq.items():
    # Calculate the idf for this word
    idf = math.log10(N / len(docs))

    # Initialize an empty dictionary to hold the tf-idf weights for this word
    tf_idf[word] = {}

    # Iterate over each document in the document frequencies
    for doc_id, freq in docs.items():
        # Calculate the tf for this word in this document
        tf = 1 + math.log10(freq)

        # Calculate the tf-idf weight for this word in this document
        tf_idf[word][doc_id] = tf * idf
# creating championslist
champion_list_size = 20  # You can adjust this value as needed

# Initialize an empty dictionary to hold the Champion Lists
champion_lists = {}

# Iterate over each word in the TF-IDF weights
for word, docs in tf_idf.items():
    # Sort the documents by their TF-IDF weight for this word
    sorted_docs = sorted(docs.items(), key=lambda item: item[1], reverse=True)

    # Get the top documents to form the Champion List for this word
    champion_lists[word] = sorted_docs[:champion_list_size]

# Now "tf_idf" is a dictionary where each key is a word, and the value is another dictionary.
# In this inner dictionary, each key is a document ID, and the value is the tf-idf weight of the word in the document.
#print("TF-IDF Weights:")
#print(tf_idf)



# Function to process the query
def process_query(query):
    # Preprocess the query in the same way as the documents
    normalized_query = normalizer.normalize(query)
    words = word_tokenize(normalized_query)
    words = [word for word in words if word not in stopwords]
    stemmed_words = [stemmer.convert_to_stem(word) for word in words]

    # Calculate the tf-idf weights for the query
    tf_idf_query = {}
    word_freq_query = Counter(stemmed_words)
    for word, freq in word_freq_query.items():
        if word in tf_idf:
            tf = 1 + math.log10(freq)
            idf = math.log10(N / len(doc_freq[word]))  # Use the IDF from the doc_freq dictionary
            tf_idf_query[word] = tf * idf

    return tf_idf_query

# Function to retrieve the top k documents
def retrieve_documents(query, k ):
    # Process the query
    tf_idf_query = process_query(query)

    # Calculate the cosine similarity for each document
    cosine_similarities = {}
    for word, docs in tf_idf_query.items():
        if word in champion_lists: #tf_idf
            for doc_id, weight in champion_lists[word]: #tf_idf[word].items()
                if doc_id not in cosine_similarities:
                    cosine_similarities[doc_id] = 0
                cosine_similarities[doc_id] += weight * docs

    # Sort the documents by cosine similarity
    sorted_docs = sorted(cosine_similarities.items(), key=lambda item: item[1], reverse=True)

    # Retrieve the top k documents
    top_k_docs = sorted_docs[:k]

    # Print the title and URL of each document
    # Test the function


    # Print the title and URL of each document
    for doc_id, similarity in top_k_docs:
        doc = next(doc for doc in processed_documents if doc['id'] == doc_id)
        print(f"Title: {doc['title']}")
        print(f"URL: {doc['url']}")
        print(f"Similarity: {similarity}")
        print()


# Test the function
# Test the function

retrieve_documents('اقدامات دولت سیزدهم' , 15)
