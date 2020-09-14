import torch
import torch.nn as nn


class KGAT(nn.Module):

    def __init__(self, args, doc_feature_dict, entity_embedding, relation_embedding, adj_entity, adj_relation):
        super(KGAT, self).__init__()
        self.args = args
        self.doc_feature_dict = doc_feature_dict
        self.entity_embedding = entity_embedding
        self.relation_embedding = relation_embedding
        self.adj_entity = adj_entity
        self.adj_relation = adj_relation
        self.entity_embedding_lookup = nn.Embedding.from_pretrained(entity_embedding)
        self.relation_embedding_lookup = nn.Embedding.from_pretrained(relation_embedding)
        self.attention_layer = nn.Linear(3*self.args.embedding_dim,1)
        self.softmax = nn.Softmax(dim=0)
        self.relu = nn.ReLU(inplace=True)
        self.convolve_layer = nn.Linear(2*self.args.embedding_dim,self.args.embedding_dim)

    def get_neighbors(self, entities):
        neighbor_entities = []
        neighbor_relations = []
        for entity_batch in entities:
            neighbor_entities.append([])
            neighbor_relations.append([])
            for entity in entity_batch:
                if type(entity) == int:
                    neighbor_entities[-1].append(self.adj_entity[entity])
                    neighbor_relations[-1].append(self.adj_relation[entity])
                else:
                    neighbor_entities[-1].append([])
                    neighbor_relations[-1].append([])
                    for entity_i in entity:
                        neighbor_entities[-1][-1].append(self.adj_entity[entity_i])
                        neighbor_relations[-1][-1].append(self.adj_relation[entity_i])

        return neighbor_entities, neighbor_relations

    def aggregate(self, entity_embedding, neighbor_embedding):
        concat_embedding = torch.cat([entity_embedding, neighbor_embedding], len(entity_embedding.shape)-1)
        aggregate_embedding = self.relu(self.convolve_layer(concat_embedding))
        return aggregate_embedding

    def forward(self, entity_ids):
        neighbor_entities, neighbor_relations = self.get_neighbors(entity_ids)

        neighbor_entity_embedding = self.entity_embedding_lookup(torch.tensor(neighbor_entities).cuda())
        neighbor_relation_embedding = self.relation_embedding_lookup(torch.tensor(neighbor_relations).cuda())
        entity_embedding = self.entity_embedding_lookup(torch.tensor(entity_ids).cuda())

        if len(entity_embedding.shape) == 3:
            entity_embedding_expand = torch.unsqueeze(entity_embedding, 2)
            entity_embedding_expand = entity_embedding_expand.expand(entity_embedding_expand.shape[0], entity_embedding_expand.shape[1], self.args.entity_neighbor_num , entity_embedding_expand.shape[3])
            embedding_concat = torch.cat([entity_embedding_expand, neighbor_entity_embedding, neighbor_relation_embedding], 3)
            attention_value = self.softmax(self.relu(self.attention_layer(embedding_concat)))
            neighbor_att_embedding = torch.sum(attention_value * neighbor_entity_embedding, dim=2)
            kgat_embedding = self.aggregate(entity_embedding, neighbor_att_embedding)
        else:
            entity_embedding_expand = torch.unsqueeze(entity_embedding, 3)
            entity_embedding_expand = entity_embedding_expand.expand(entity_embedding_expand.shape[0], entity_embedding_expand.shape[1],entity_embedding_expand.shape[2], self.args.entity_neighbor_num , entity_embedding_expand.shape[4])
            embedding_concat = torch.cat([entity_embedding_expand, neighbor_entity_embedding, neighbor_relation_embedding], 4)
            attention_value = self.softmax(self.relu(self.attention_layer(embedding_concat)))
            neighbor_att_embedding = torch.sum(attention_value * neighbor_entity_embedding, dim=3)
            kgat_embedding = self.aggregate(entity_embedding, neighbor_att_embedding)

        return kgat_embedding