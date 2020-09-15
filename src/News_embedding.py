import torch
import torch.nn as nn
from KGAT import KGAT


class News_embedding(nn.Module):

    def __init__(self, args, doc_feature_dict, entity_embedding, relation_embedding, adj_entity, adj_relation, entity_num, position_num, type_num):
        super(News_embedding, self).__init__()
        self.args = args
        self.doc_feature_dict = doc_feature_dict
        self.entity_embedding = entity_embedding
        self.relation_embedding = relation_embedding
        self.adj_entity = adj_entity
        self.adj_relation = adj_relation
        self.kgat = KGAT(args, doc_feature_dict, entity_embedding, relation_embedding, adj_entity, adj_relation)
        self.entity_embedding_lookup = nn.Embedding.from_pretrained(entity_embedding)

        self.entity_num = entity_num
        self.position_num = position_num
        self.type_num = type_num

        self.final_embedding = nn.Linear(2*self.args.embedding_dim,self.args.embedding_dim)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        self.title_embeddings = nn.Embedding(2, self.args.embedding_dim)
        self.type_embeddings = nn.Embedding(4, self.args.embedding_dim)
        self.entity_num_embeddings = nn.Embedding(entity_num, self.args.embedding_dim)

        # Use xavier initialization method to initialize embeddings of entities and relations
        title_weight = torch.FloatTensor(self.position_num, self.args.embedding_dim).cuda()
        type_weight = torch.FloatTensor(self.type_num, self.args.embedding_dim).cuda()
        entity_num_weight = torch.FloatTensor(entity_num, self.args.embedding_dim).cuda()

        nn.init.xavier_normal_(title_weight, gain=0.01)
        nn.init.xavier_normal_(type_weight, gain=0.01)
        nn.init.xavier_normal_(entity_num_weight, gain=0.01)

        self.title_embeddings.weight = nn.Parameter(title_weight)
        self.type_embeddings.weight = nn.Parameter(type_weight)
        self.entity_num_embeddings.weight = nn.Parameter(entity_num_weight)

        self.attention_embedding_layer1 = nn.Linear(2*self.args.embedding_dim,self.args.layer_dim)
        self.attention_embedding_layer2 = nn.Linear(self.args.layer_dim,1)
        self.softmax = nn.Softmax(dim=-2)


    def attention_layer(self, entity_embeddings, context_vecs):
        if len(entity_embeddings.shape) == 4:
            context_vecs = torch.unsqueeze(context_vecs, -2)
            context_vecs = context_vecs.expand(context_vecs.shape[0], context_vecs.shape[1], entity_embeddings.shape[2], context_vecs.shape[3])
        else:
            context_vecs = torch.unsqueeze(context_vecs, -2)
            context_vecs = context_vecs.expand(context_vecs.shape[0], entity_embeddings.shape[1], context_vecs.shape[2])

        att_value1 = self.relu(self.attention_embedding_layer1(torch.cat([entity_embeddings, context_vecs], dim=-1)))
        att_value = self.relu(self.attention_embedding_layer2(att_value1))
        soft_att_value = self.softmax(att_value)
        weighted_entity_embedding = soft_att_value*entity_embeddings
        weighted_entity_embedding_sum = torch.sum(weighted_entity_embedding, dim=-2)
        return weighted_entity_embedding_sum


    def get_entities_ids(self, news_id):
        entities = []
        for news in news_id:
            if type(news) == str:
                entities.append(self.doc_feature_dict[news][0])
            else:
                entities.append([])
                for news_i in news:
                    entities[-1].append(self.doc_feature_dict[news_i][0])
        return entities

    def get_entities_nums(self, news_id):
        entities_nums = []
        for news in news_id:
            if type(news) == str:
                entities_nums.append(self.doc_feature_dict[news][1])
            else:
                entities_nums.append([])
                for news_i in news:
                    entities_nums[-1].append(self.doc_feature_dict[news_i][1])
        return entities_nums

    def get_istitles(self, news_id):
        istitles = []
        for news in news_id:
            if type(news) == str:
                istitles.append(self.doc_feature_dict[news][2])
            else:
                istitles.append([])
                for news_i in news:
                    istitles[-1].append(self.doc_feature_dict[news_i][2])
        return istitles

    def get_type(self, news_id):
        istopics = []
        for news in news_id:
            if type(news) == str:
                istopics.append(self.doc_feature_dict[news][3])
            else:
                istopics.append([])
                for news_i in news:
                    istopics[-1].append(self.doc_feature_dict[news_i][3])
        return istopics

    def get_context_vector(self, news_id):
        context_vectors = []
        for news in news_id:
            if type(news) == str:
                context_vectors.append(self.doc_feature_dict[news][4])
            else:
                context_vectors.append([])
                for news_i in news:
                    context_vectors[-1].append(self.doc_feature_dict[news_i][4])
        return context_vectors

    def get_entity_num_embedding(self, entity_nums):
        entity_num_embedding = self.entity_num_embeddings(torch.tensor(entity_nums).cuda())
        return entity_num_embedding

    def get_title_embedding(self, istitles):
        istitle_embedding = self.title_embeddings(torch.tensor(istitles).cuda())
        return istitle_embedding

    def get_type_embedding(self, type):
        type_embedding = self.type_embeddings(torch.tensor(type).cuda())
        return type_embedding


    def forward(self, news_id):
        entities = self.get_entities_ids(news_id)
        entity_nums = self.get_entities_nums(news_id)
        istitle = self.get_istitles(news_id)
        type = self.get_type(news_id)
        context_vecs = self.get_context_vector(news_id)

        entity_num_embedding = self.get_entity_num_embedding(entity_nums)
        istitle_embedding = self.get_title_embedding(istitle)
        type_embedding = self.get_type_embedding(type)
        kgat_entity_embeddings = self.kgat(entities)  # batch(news num) * entity num
        news_entity_embedding = kgat_entity_embeddings + entity_num_embedding + istitle_embedding + type_embedding #todo

        aggregate_embedding = self.attention_layer(news_entity_embedding, torch.FloatTensor(context_vecs).cuda())

        concat_embedding = torch.cat([aggregate_embedding, torch.FloatTensor(context_vecs).cuda()],
                                     len(aggregate_embedding.shape) - 1)
        news_embeddings = self.tanh(self.final_embedding(concat_embedding))

        return news_embeddings