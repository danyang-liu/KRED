## Model description

![](./framework.PNG)


##  data description

You need to prepare three parts of data for this experiment. We offer a toy example dataset for running the code. Each items in files
 are separated with tab.

1 **Graph related data**, including a graph file, an entity vector file, a relation vector file and an entity2id file.
In the graph file, we already changed entities and relations to ids and you can get the vectors in vector files by ids
(e.g. entityid:k is the k<sup>th</sup> vector in entityvec file):
- graph.tsv: triples of knowledge graph:  head, relation, tail
- entity2vec.vec: entity embedding from news graph triples using [TransE](https://github.com/thunlp/Fast-TransX), each line 
is an entity's embedding vector and we can match it to entity according to its line number, so we don't have entityid in this file.
- relation2vec.vec relation embedding same with entity embedding.
- entity2id.txt: entity, entityid
- relation2id.txt

2 **Train and test data**, the dataset is splited by time.
- train.tsv: train file including userid, docid, label
- test.tsv: val file including sessionid, userid, docid, label

3 **User and document data**, including users' click history file and document feature file. For document feature, 
we have the document vector and features of each entities in this document. The format of entity feature is 
entityid, frequency, position, type of this entity and separated with space. Entities in document feature file should be included in entity2id file.
- click_history.tsv: userid, docids
- doc_feature.tsv: docid, entity_features, document vector

## Multi task training

step1: select training method: single task training or multi task training

step2: select task, we have 5 core news recommendation tasks, including: user2item recommendation task, item2item recommendation task, 

##  run the code

python main.py 

## environment
yaml
