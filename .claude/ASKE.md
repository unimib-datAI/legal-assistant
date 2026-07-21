# ASKE (Automated System for Knowledge Extraction)

## Data Preparation
The input of the system consist in the following parts:
* A corpus D of legal documents that must be preprocessed with the following steps:
  * Tokenization: split document chunks into individual parts
  * Lemmatization: reduce words to their base form
* Transform document chunks and the seed concept in their corresponding vector representation, projecting them in 
the same semantic space using a SentenceTransformer model (e.g., `all-MiniLM-L6-v2`).

NB: A document chunk should correspond to a logical unit of the document (such as a single sentence or a paragraph) whose length
can provide enough context for its practical usage; moreover it must fit the fixed size window of the embedding model 
used in the system (e.g., 512 tokens for `all-MiniLM-L6-v2`).
to achieve the first run it is possible to rely on a simple keyword seed (like our approach) or with a textual description of
the concept.

## Document Chunk Classification 
In this phase ASKE performs a zero-shot classification of document chunks that will be assigned to zero or some concepts;
a similarity measure (cosine similarity) is computed between the vector embeddings $\bar{k}$ of each **document chunk** 
and the embeddings of each **concept** $\bar{c}$.

$f(k, \mathcal{C}) = \{c_i \in C : \sigma(\bar{k}, \bar{c}_i) \geq \alpha\}$

A threshold α is defined to determine the minimum similarity score required for a chunk to be classified under a concept; 
if the similarity score between a chunk and a concept exceeds α, the chunk is assigned to that concept. (The threshold has
been chosen following the benchmark of the paper, in order to achieve the best results possible).
Lastly, if a concept does not classify any document chunk it will be deactivated.

## Terminology Enrichment
For each concept $c_i$, we first extract the classified document chunk by their index;

Chunk indexes classified under a list of concepts:
```json
{
  "redress and complaint mechanism": [1,5, 7],
  "lawfulness fairness and transparency of processing": [2, 3, 8],
  ...
}
```
Chunks extracted from the current concept $c_i$ (example): "lawfulness fairness and transparency of processing"
```python
classified_chunks_by_current_concept = [
'this regulation lay down rule relating to the protection of natural person with regard to the processing of personal data and rule relating to the free movement of personal data .',
'this regulation protects fundamental right and freedom of natural person and in particular their right to the protection of personal data .',
'the free movement of personal data within the union shall be neither restricted nor prohibited for reason connected with the protection of natural person with regard to the processing of personal data .',
  # And more chunks...
]
```
Then a set of terms $W_i$ sorted by their frequency are extracted from the chunks:
```python
candidate_terms = ['data', 'controller', 'processing', 'processor', 'protection', 'measure', 'authority', 'state', 'member', 
                   'interest', 'category', 'obligation', 'freedom', 'person', 'case', 'risk', 'account', 'organisation', 
                   'safeguard', 'security', 'operation', 'mean', 'scope', 'assessment', 'officer', 'transfer', 'restriction']
```

For each of these terms $W_i$ we extract the corresponding definition $W_d$ retrieved from the external dictionary (e.g., WordNet)
```python
terms_definition = [
  {
    'definition': 'a collection of facts from which conclusions may be drawn',
    'label': 'data'
  },
  {
    'definition': 'someone who maintains and audits business accounts',
    'label': 'controller'
  },
  {
    'definition': 'preparing or putting through a prescribed procedure',
    'label': 'processing'
  }
]
```
All the terms definition will be embedded with the same SentenceTransformer model used in the previous phases, 
and the similarity between the term definition and the classified document chunks will be computed, sorted by their similarity score
and the top N terms will be selected as candidate terms for the current concept $c_i$ by the parameter gamma (e.g., 0.5).

## Concept Derivation
