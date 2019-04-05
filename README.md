# SDML  
Special Topics in Machine Learning (SDML) at NTU 2018 Fall.
This repo includes codes of projects in the SDML course, NTU. Only some of them are shown here:
   1. Graph Embedding
   2. Recommendation System
   3. Controlled Text Generation 
   4. Clustering (not included in this repo)
   5. Real Time Bidding (not included in this repo)
   
### HW1-1   
* Task: 
  * Link Prediction
* Feature: 
  * Graph (node id + edge)
* method: 
  * DeepWalk, Node2Vec, Graph Convolutional Networks
  * Cosine Similarity ----> Threshold

### HW1-2   
* Task: 
  * link prediction
* Feature: 
  * Graph (node id + edge), Title, Abstract
* method: 
  * DeepWalk, Node2Vec, Graph Convolutional Networks
  * Cosine Similarity, Word Embedding, Graph Features (jaccard distacne, degree..) ----> Classifiy

### HW1-3  
* Task: 
  * Link Prediction
* Feature: 
  * Graph (node id + edge), Title, Abstract, Time
* method:
  * Nearly the same to HW1-2
  * Auto Feature Generation and Selection

### HW2
* Task: 
  * Food Recomendation (no repeat)
* Feature: 
  * User Rating
* method: 
  * WARP loss
  * Cocurrence Matrix ----> DeepFM
