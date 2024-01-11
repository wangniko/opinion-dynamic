import numpy as np
import pandas as pd
import networkx as nx
from math_utils import scaling
from ax import optimize
import collections

class Netizen(object):
    def __init__(self, netizen_size, probability =0.4, nweight_lb = 0.1,nweight_ub = 0.5) -> None:
        """
        initialize netizens' belif and network
        
        """
        self.netizen_size = netizen_size
        self.probability = probability
        self.nweight_lb = nweight_lb
        self.nweight_ub = nweight_ub

    def generate_netizen_network(self):
        '''
        Generate graph for simple netizen network.
        :param netizen_size: number of netizen nodes
        :param p: probability for edge creation
        :return: edge info transformed from networkx Graph object
        '''
        G_netizen = nx.Graph()
        nweight = np.random.uniform(self.nweight_lb,self.nweight_ub)
        for u,v in nx.erdos_renyi_graph(n = self.netizen_size, p = self.probability, directed=False).edges():
            G_netizen.add_edge(u,v,weight = nweight)
            df_netizen = nx.to_pandas_edgelist(G_netizen)
        df_netizen.source = df_netizen.source.map(lambda x: 'N'+ str(x))
        df_netizen.target = df_netizen.target.map(lambda x: 'N'+ str(x))
        return df_netizen


class Media(Netizen):
    def __init__(self, netizen_size, probability, nweight_lb,nweight_ub,
                 media_size, rate, reputation, heat, mweight_lb, mweight_ub):
        super().__init__(netizen_size, probability, nweight_lb,nweight_ub)
        """
        Interaction between medias and netizens
        :param media_size: number of media nodes
        :param reputation: reputation of each media.
        :param heat: heat of each media.
        :param mweight_lb: the lower bound of the edge weight between medias and netizens.
        :param mweight_ub: the upper bound of the edge weight between medias and netizens.
        """
        self.media_size = media_size
        self.reputation = reputation
        self.rate = rate
        self.heat = heat
        self.mweight_lb = mweight_lb
        self.mweight_ub = mweight_ub
        self.media_id = ['M'+ str(i) for i in range(self.media_size)]
        self.netizen_id = ['N'+ str(i) for i in range(self.netizen_size)]

    def media_followers_network(self):
        """
        Create interation graph based on the media exposure rate
        :param rate: exposure rate of each media.
        return a dataframe describing media_netizen relationships
        """
        media_followers_counts =  np.multiply(np.repeat(self.netizen_size,self.media_size), self.rate)
        media_trust = scaling(self.reputation, self.mweight_lb, self.mweight_ub)
        media_id = ['M'+ str(i) for i in range(self.media_size)]
        sample_netizens = {}
        media_followers = []
        for index, value in enumerate(media_id):
            sample_netizens[value] = np.random.choice(self.netizen_size,int(media_followers_counts[index]), replace = False)
            for netizen in sample_netizens[value]:
                media_followers.append(( value, 'N'+ str(netizen), media_trust[index]))
        df_media = pd.DataFrame(media_followers, columns = ['source', 'target','weight'])    
        return df_media


    def netizen_belif_update(self, belif, report):
        """
        exhibit netizens' belif updating process affected by medias and peers
        return netizens' belif after updating
        """
        df_media = self.media_followers_network()
        media_id = list(set(df_media.source.to_list()))
        netizen_id  = list(set(df_media.target.to_list()))
        df_attributes1 = pd.DataFrame({'id': media_id + netizen_id,
                                       'type': np.repeat(1,len(media_id)).tolist() + np.zeros(len(netizen_id)).tolist()})
        belif_dict = dict(zip(['N'+ str(i) for i in range(self.netizen_size)], belif))
        report_dict = dict(zip(['M'+ str(i) for i in range(self.media_size)], report))

        status_dict = dict(belif_dict,**report_dict)
        df_attributes1['status'] = df_attributes1['id'].map(status_dict)

        
        # 分别给网民和媒体节点添加属性，网民节点属性为belif, 媒体节点属性为report
        node_attributes1 = df_attributes1.set_index('id').to_dict('index')
        G_media = nx.from_pandas_edgelist(df_media, edge_attr= True, create_using=nx.Graph())
        G_media.remove_edges_from(nx.selfloop_edges(G_media))
        nx.set_node_attributes(G_media, node_attributes1)

        netizen_status_update = {}
        # 首先考虑网民与媒体的交互之后的信念更新
        for node in G_media.nodes():
            if G_media.nodes[node]['type'] == 0:
                if len(list(nx.neighbors(G_media, node))) == 1:
                    trust_media = list(nx.neighbors(G_media, node))[0]
                    trust = G_media[trust_media][node]["weight"]
                    G_media.nodes[node]['status'] = G_media.nodes[node]['status'] + trust*abs(G_media.nodes[node]['status'] - G_media.nodes[trust_media]['status'])
                elif len(list(nx.neighbors(G_media, node))) > 1:
                    weight_list = [G_media[media][node]["weight"] for media in list(nx.neighbors(G_media,node))]
                    belif_delta = []
                    for media in list(nx.neighbors(G_media, node)):
                        belif_delta.append(G_media[media][node]["weight"]* abs(G_media.nodes[node]['status'] - G_media.nodes[media]['status']))
                    G_media.nodes[node]['status'] = G_media.nodes[node]['status'] + np.sum(np.array(belif_delta))/np.sum(np.array(weight_list))
                else:
                    continue
                netizen_status_update[node] = G_media.nodes[node]['status']
            else:
                continue
            

        # 网民之间互动信念更新
        belif_dict.update(netizen_status_update)

        df_netizen = super().generate_netizen_network()
        G_netizen = nx.from_pandas_edgelist(df_netizen, edge_attr= True)
        G_netizen.remove_edges_from(nx.selfloop_edges(G_netizen))
        df_attributes2 = pd.DataFrame({'id': ['N'+ str(i) for i in range(self.netizen_size)],
                                       'status': belif})
        df_attributes2['status'] = df_attributes2['id'].map(belif_dict)

        node_attributes2 = df_attributes2.set_index('id').to_dict('index')
        nx.set_node_attributes(G_netizen, node_attributes2)

        for i in range(0, nx.number_of_nodes(G_netizen)):
            # 随机选择一个网民
            n1 = list(G_netizen.nodes())[np.random.randint(0, nx.number_of_nodes(G_netizen))]

            # 找到该网民的邻居
            neighbours = list(nx.neighbors(G_netizen, n1))
            if len(neighbours) == 0:
                continue

            # 随机选一个邻居
            n2 = neighbours[np.random.randint(0, len(neighbours))]

            # 更新该网民和其邻居的信念
            G_netizen.nodes[n1]['status'] = G_netizen.nodes[n1]['status'] + G_netizen[n1][n2]["weight"] * abs(G_netizen.nodes[n1]['status'] - G_netizen.nodes[n2]['status'])
            G_netizen.nodes[n2]['status'] = G_netizen.nodes[n2]['status'] + G_netizen[n1][n2]["weight"] * abs(G_netizen.nodes[n1]['status'] - G_netizen.nodes[n2]['status'])

        ## 输出网民更新后的信念
        netizen_belif_u2 = {}
        for node in G_netizen.nodes():
            netizen_belif_u2[node] = G_netizen.nodes[node]['status']

        belif_dict.update(netizen_belif_u2)
        netizen_belif_new = np.array(list(collections.OrderedDict(sorted(belif_dict.items(), key=lambda t: t[0])).values()))

        return netizen_belif_new 
    
    ## 定义网民对于媒体的反应动作是显式更新的，且是公共知识

    def followers_response(self,  belif, report, lb, ub):
        df_media = self.media_followers_network()
        followers_response = {}
        ## 首先获得网民与媒体和同辈互动后更新的信念
        netizen_belif_new = self.netizen_belif_update(belif, report)
        belif_dict =  dict(zip(['N'+ str(i) for i in range(self.netizen_size)], netizen_belif_new))
        report_dict = dict(zip(['M'+ str(i) for i in range(self.media_size)], report))

        for keys, values in  report_dict.items():
            pos, neg, neutral = 0, 0, 0
            followers_response[keys] = {}
            media_followers = df_media.loc[df_media['source'] == keys, 'target'].tolist()
            media_followers_belif = [belif_dict[netizen] for netizen in media_followers]
            for belif in media_followers_belif:
                if abs(belif - values) < lb:
                    pos +=1
                elif abs(belif - values) > ub:
                    neg += 1
                else:
                    neutral += 1
            followers_response[keys]['pos'] = pos
            followers_response[keys]['neg'] = neg
            followers_response[keys]['neutral'] = neutral
        return followers_response

    # 定义中间函数计算媒体的效用感知

    def media_performance_t_response(self, belif, report, pos, neg, propensity, cost, lb, ub):
        followers_response = self.followers_response( belif, report, lb, ub)
        media_pos_reponses = [followers_response[keys]['pos'] for keys, values in followers_response.items()]
        media_pos_reponses = [followers_response[keys]['neg'] for keys, values in followers_response.items()]

        return np.multiply(propensity, np.multiply(pos, media_pos_reponses)) + np.multiply(np.ones(self.media_size) - propensity, np.multiply(neg, media_pos_reponses)) - cost
    

    def func(self, netizen_belif_t_1, media_report_t_1,i,p,pos, neg, propensity, cost, lb, ub):
        media_report_opt = dict(zip(['M'+ str(i) for i in range(self.media_size)], media_report_t_1))
        media_report_opt[i] = p["report"]
        media_performance = self.media_performance_t_response(netizen_belif_t_1, media_report_t_1, pos, neg, propensity, cost, lb, ub)
        media_performance_opt = dict(zip(['M'+ str(i) for i in range(self.media_size)], media_performance))
        return media_performance_opt[i]

    
    def media_action(self, netizen_belif_t_1, media_report_t_1,pos, neg, propensity, cost, lb, ub):
        media_id = ['M'+ str(i) for i in range(self.media_size)]  
        media_report_t = {}
        for i in media_id:
            best_parameters, best_values, experiment, model = optimize(
                parameters=[
                {
                        "name": "report",
                        "type": "range",
                        "bounds": [0, 1],
                }
                ],
                # Booth function
                evaluation_function = lambda p: self.func( netizen_belif_t_1, media_report_t_1,i,p,pos, neg, propensity, cost, lb, ub),
                minimize = False,
                total_trials = 2
            )
            media_report_t[i] = best_parameters['report']
        return np.array(list(media_report_t.values()))


    def media_state_update(self,netizen_belif_t_1, media_report_t_1,pos, neg, propensity, cost, lb, ub):
        media_report_t = self.media_action(netizen_belif_t_1, media_report_t_1,pos, neg, propensity, cost, lb, ub)
        netizen_belif_t = self.netizen_belif_update(netizen_belif_t_1, media_report_t)
        followers_response = self.followers_response(netizen_belif_t, media_report_t,  lb, ub)
        media_pos_reponses = [followers_response[keys]['pos'] for keys, values in followers_response.items()]
        media_neg_reponses = [followers_response[keys]['neg'] for keys, values in followers_response.items()]

        media_heat_t = self.heat +  media_pos_reponses
        media_reputation_t = self.reputation - media_neg_reponses
        print(media_reputation_t,media_heat_t,netizen_belif_t,media_report_t)
        return scaling(media_reputation_t, 0, 1000), scaling(media_heat_t, 0, 1000), scaling(netizen_belif_t, 0, 1), media_report_t

