# %%
# created by cshyun102@gmail.com

import copy
from copy import deepcopy
import pandas as pd
import numpy as np
import random


class PROCLUS_SPMMP:
    def __init__(
        self, transaction_df, k, l, B=5, minDeviation=0.1, max_iteration=10
    ):
        from sklearn.preprocessing import OneHotEncoder

        self.transaction_df = transaction_df
        self.dimensions = transaction_df.columns.tolist()
        self.domains = {
            dimension: transaction_df[dimension].unique().tolist()
            for dimension in self.dimensions
        }

        enc = OneHotEncoder()
        enc.fit(transaction_df)
        attributes = enc.categories_
        encoded_array = enc.transform(transaction_df).toarray()
        columns = []
        for i in attributes:
            columns = columns + i.tolist()
        one_hot_encoded_df = pd.DataFrame(
            encoded_array, columns=columns, index=transaction_df.index
        )
        self.one_hot_df = one_hot_encoded_df
        self.indice = transaction_df.index.tolist()
        self.k = k
        self.l = l
        self.minDeviation = minDeviation
        self.B = B
        self.max_iteration = max_iteration

        delta_distance_dict = {}
        for i in self.dimensions:
            delta_distance_dict[i] = {}
        self.delta_distance_dict = delta_distance_dict

        print("end __init__")

    # 모든 점은 index를 함수의 input으로 활용한다.
    def distance_on_single_dimension(self, index_1, index_2, dimension):
        """
        # TODO: Centroid 계산과 동일하도록
        # if (
        #     frozenset((index_1, index_2))
        #     in self.delta_distance_dict[dimension].keys()
        # ):
        #     return self.delta_distance_dict[dimension][
        #         frozenset((index_1, index_2))
        #     ]
        # else:
        #     point_1 = self.one_hot_df.loc[index_1][self.domains[dimension]]
        #     point_2 = self.one_hot_df.loc[index_2][self.domains[dimension]]
        #     distance = np.mean(np.power(point_1 - point_2, 2))
        #     self.delta_distance_dict[dimension][
        #         frozenset((index_1, index_2))
        #     ] = distance
        #     return distance
        """
        point_1 = self.transaction_df.loc[index_1][dimension]
        point_2 = self.transaction_df.loc[index_2][dimension]
        if point_1 == point_2:
            return 0
        else:
            return 1

    def distance_on_multi_dimensions(self, index_1, index_2, dimension_list):
        """
        # TODO: Centroid 계산과 동일하도록
        point_1 = self.one_hot_df.loc[index_1]
        point_2 = self.one_hot_df.loc[index_2]
        minus_series = point_1 - point_2
        distance_list = []
        for dimension in dimension_list:
            distance_list.append(
                self.distance_on_single_dimension(index_1, index_2, dimension)
                # np.mean(np.power(minus_series[self.domains[dimension]], 2))
            )
        return np.mean(distance_list)
        """
        point_1 = self.transaction_df.loc[index_1][dimension_list]
        point_2 = self.transaction_df.loc[index_2][dimension_list]
        return np.mean(point_1 != point_2)

    def Greedy(self):
        num_medoids = self.k * self.B
        if num_medoids > len(self.indice):
            num_medoids = len(self.indice)
        medoids = []
        sample = copy.deepcopy(self.indice)

        m1 = random.choice(sample)
        sample.remove(m1)
        medoids.append(m1)

        distance_dict = {}
        for x in sample:
            distance_dict[x] = self.distance_on_multi_dimensions(
                m1, x, self.dimensions
            )

        for k in range(num_medoids - 1):
            # for x in sample:
            #     for i in medoids:
            #         distance_dict[x] = min(
            #             distance_dict[x],
            #             self.distance_on_multi_dimensions(
            #                 i, x, self.dimensions
            #             ),
            #         )
            mk = max(distance_dict, key=distance_dict.get)
            medoids.append(mk)
            sample.remove(mk)
            del distance_dict[mk]
            for x in sample:
                distance_dict[x] = min(
                    distance_dict[x],
                    self.distance_on_multi_dimensions(mk, x, self.dimensions),
                )
        return medoids

    def FindLocality(self, medoids):
        delta_dict = {}
        for i in medoids:
            min_distance = np.inf
            for j in medoids:
                if i != j:
                    if min_distance > self.distance_on_multi_dimensions(
                        i, j, self.dimensions
                    ):
                        min_distance = self.distance_on_multi_dimensions(
                            i, j, self.dimensions
                        )
            delta_dict[i] = min_distance

        sample = copy.deepcopy(self.indice)
        for i in medoids:
            sample.remove(i)

        Locality_dict = {}
        for i in medoids:
            Locality_dict[i] = []
            for j in sample:
                if (
                    self.distance_on_multi_dimensions(i, j, self.dimensions)
                    < delta_dict[i]
                ):
                    Locality_dict[i].append(j)
        return Locality_dict

    def FindDimensions(self, medoids, Locality_dict):
        k = self.k
        l = self.l
        X_ij = {}
        for i in medoids:
            X_ij[i] = {}
            for j in self.dimensions:
                distance_list = []
                for x in Locality_dict[i]:
                    distance_list.append(
                        self.distance_on_single_dimension(i, x, j)
                    )
                X_ij[i][j] = np.mean(distance_list)
        Y_i = {}
        sigma_i = {}
        Z_ij = {}
        for i in medoids:
            Y_i[i] = np.sum([X_ij[i][j] for j in self.dimensions]) / len(
                self.dimensions
            )
            sigma_i[i] = np.sqrt(
                np.sum([(X_ij[i][j] - Y_i[i]) ** 2 for j in self.dimensions])
                / (len(self.dimensions) - 1)
            )
            Z_ij[i] = {}
            for j in self.dimensions:
                if sigma_i[i] != 0:
                    Z_ij[i][j] = (X_ij[i][j] - Y_i[i]) / sigma_i[i]
                else:
                    Z_ij[i][j] = np.inf

        Di = {}
        picked_list = []  # picked_list에는 (i,j) 순으로 넣는다

        for i in medoids:
            temp_dimension_list = []
            temp_Z_list = []
            for j in self.dimensions:
                temp_dimension_list.append(j)
                temp_Z_list.append(Z_ij[i][j])
            temp_sort = [
                key
                for key, value in sorted(
                    zip(temp_dimension_list, temp_Z_list),
                    key=lambda pair: pair[1],
                )
            ]
            # 반드시 2개의 dimension은 포함하도록 한다.
            for index in range(2):
                picked_list.append((i, temp_sort[index]))

        total_dimension_list = []
        total_Z_list = []
        for i in medoids:
            for j in self.dimensions:
                total_dimension_list.append((i, j))
                total_Z_list.append(Z_ij[i][j])
        total_sort = [
            key
            for key, value in sorted(
                zip(total_dimension_list, total_Z_list),
                key=lambda pair: pair[1],
            )
        ]

        while len(picked_list) < k * l:
            target_dimension = total_sort[0]
            if target_dimension not in picked_list:
                picked_list.append(target_dimension)
            del total_sort[0]

        for picked_dimension in picked_list:
            if picked_dimension[0] not in Di.keys():
                Di[picked_dimension[0]] = [picked_dimension[1]]
            else:
                Di[picked_dimension[0]].append(picked_dimension[1])
        return Di

    def AssignPoints(self, medoids, Di):
        sample = copy.deepcopy(self.indice)
        cluster_dict = {}

        for i in medoids:
            sample.remove(i)
            cluster_dict[i] = [i]

        distance_dict = {}

        for p in sample:
            distance_dict[p] = np.inf
            for i in medoids:
                temp_distance = self.distance_on_multi_dimensions(i, p, Di[i])
                if temp_distance < distance_dict[p]:
                    distance_dict[p] = temp_distance
                    min_medoid = i
            cluster_dict[min_medoid].append(p)
        return cluster_dict

    def EvaluateClusters(self, medoids, Di, cluster_dict):
        # TODO: CENTROID 기준으로 바꿀 필요 있음.
        Y_ij = {}
        w_i = {}

        CENTROID_i = {}

        # TODO: CENTROID 중심 계산법을 활용함.
        for i in medoids:
            target_one_hot_df = self.one_hot_df.loc[cluster_dict[i]]
            CENTROID_i[i] = target_one_hot_df.mean(axis=0)
            Y_ij[i] = {}
            for j in Di[i]:
                centroid_dimension_one_hot = CENTROID_i[i][self.domains[j]]
                target_dimension_one_hot = target_one_hot_df[self.domains[j]]
                Y_ij[i][j] = (
                    (
                        target_dimension_one_hot.sub(
                            centroid_dimension_one_hot
                        )
                        .abs()
                        .to_numpy()
                        .sum()
                    )
                    / 2
                    / len(target_dimension_one_hot)
                )

            w_i[i] = np.mean([Y_ij[i][j] for j in Di[i]])

        return np.sum(
            [w_i[i] * len(cluster_dict[i]) for i in medoids]
        ) / np.sum([len(cluster_dict[i]) for i in medoids])

    def Initiliazation(self):
        self.M = self.Greedy()
        print("END intialize")

    def Iterative(self):
        # 이 코드는 agent를 5개를 두고, 반은 수렴하고, 반은 랜덤하게 간다.
        agent_dict = {}
        num_agent = 10
        for i in range(num_agent):
            agent_dict[i] = {}
            agent_dict[i]["medoids"] = random.sample(self.M, self.k)
            agent_dict[i]["isbest"] = False
            agent_dict[i]["sample"] = copy.deepcopy(self.M)

        BestObjective = np.inf
        converge_count = 0
        while converge_count < self.max_iteration:
            objective_change = False
            for i in range(num_agent):
                if agent_dict[i]["isbest"] is False:
                    agent_dict[i]["locality"] = self.FindLocality(
                        agent_dict[i]["medoids"]
                    )
                    agent_dict[i]["Di"] = self.FindDimensions(
                        agent_dict[i]["medoids"], agent_dict[i]["locality"]
                    )
                    agent_dict[i]["cluster"] = self.AssignPoints(
                        agent_dict[i]["medoids"], agent_dict[i]["Di"]
                    )
                    agent_dict[i]["objective"] = self.EvaluateClusters(
                        agent_dict[i]["medoids"],
                        agent_dict[i]["Di"],
                        agent_dict[i]["cluster"],
                    )

            for i in range(num_agent):
                if agent_dict[i]["objective"] < BestObjective:
                    BestObjective = agent_dict[i]["objective"]
                    M_best = agent_dict[i]["medoids"]
                    locality_dict_best = agent_dict[i]["locality"]
                    Di_best = agent_dict[i]["Di"]
                    cluster_dict_best = agent_dict[i]["cluster"]
                    agent_dict[i]["isbest"] = True
                    agent_dict[i]["sample"] = copy.deepcopy(self.M)

                    objective_change = True
                    best_agent_index = i

                    converge_count = 0
                    print("BEST OBJECTIVE:", BestObjective)
            if objective_change is True:
                for i in range(num_agent):
                    agent_dict[i]["sample"] = copy.deepcopy(self.M)
                    for j in M_best:
                        agent_dict[i]["sample"].remove(j)
                    if i != best_agent_index:
                        agent_dict[i]["isbest"] = False

            if objective_change is False:
                converge_count += 1
                print("BEST OBJECTIVE:", BestObjective)

            # TODO: BAD medoids 의 기준을 단순히 숫자가 아닌 다른 것으로 할 수 있는가?
            bad_medoids = []
            for medoid in M_best:
                if len(cluster_dict_best[medoid]) < len(
                    self.indice
                ) * self.minDeviation / len(M_best):
                    bad_medoids.append(medoid)

            if len(bad_medoids) != 0:
                print("bad medoids found:", len(bad_medoids))

            for i in range(num_agent):
                if i <= 0.5 * num_agent:
                    if len(bad_medoids) != 0:
                        if len(agent_dict[i]["sample"]) > len(bad_medoids):
                            agent_dict[i]["medoids"] = copy.deepcopy(M_best)
                            for medoid in bad_medoids:
                                agent_dict[i]["medoids"].remove(medoid)
                                change_index = random.choice(
                                    agent_dict[i]["sample"]
                                )
                                agent_dict[i]["sample"].remove(change_index)
                                agent_dict[i]["medoids"].append(change_index)
                        else:
                            agent_dict[i]["medoids"] = random.sample(
                                self.M, self.k
                            )
                            agent_dict[i]["sample"] = copy.deepcopy(self.M)
                            for x in M_best:
                                agent_dict[i]["sample"].remove(x)
                    else:
                        if len(agent_dict[i]["sample"]) > 0:
                            print("bad medoids not found")
                            agent_dict[i]["medoids"] = copy.deepcopy(M_best)
                            agent_dict[i]["medoids"].remove(
                                random.choice(agent_dict[i]["medoids"])
                            )
                            change_index = random.choice(
                                agent_dict[i]["sample"]
                            )
                            agent_dict[i]["sample"].remove(change_index)
                            agent_dict[i]["medoids"].append(change_index)
                        else:
                            agent_dict[i]["medoids"] = random.sample(
                                self.M, self.k
                            )
                            agent_dict[i]["sample"] = copy.deepcopy(self.M)
                            for x in M_best:
                                agent_dict[i]["sample"].remove(x)
                else:
                    if len(bad_medoids) != 0:
                        if len(agent_dict[i]["sample"]) > len(bad_medoids):
                            agent_dict[i]["medoids"] = copy.deepcopy(M_best)
                            for medoid in bad_medoids:
                                agent_dict[i]["medoids"].remove(medoid)
                                change_index = random.choice(
                                    agent_dict[i]["sample"]
                                )
                                agent_dict[i]["sample"].remove(change_index)
                                agent_dict[i]["medoids"].append(change_index)
                        else:
                            agent_dict[i]["medoids"] = random.sample(
                                self.M, self.k
                            )
                            agent_dict[i]["sample"] = copy.deepcopy(self.M)
                            for x in M_best:
                                agent_dict[i]["sample"].remove(x)
                    else:
                        print("bad medoids not found")
                        agent_dict[i]["medoids"] = random.sample(
                            self.M, self.k
                        )
                        agent_dict[i]["sample"] = copy.deepcopy(self.M)
                        for x in M_best:
                            agent_dict[i]["sample"].remove(x)

        self.medoids = M_best
        self.locality_dict = locality_dict_best
        self.Di = Di_best
        self.cluster_dict = cluster_dict_best
        self.objective = BestObjective
        print("end iterative")

    def Refinement(self):
        dimension_dict = self.FindDimensions(self.medoids, self.cluster_dict)
        cluster_dict = self.AssignPoints(self.medoids, dimension_dict)
        Delta_dict = {}
        for i in self.medoids:
            Delta_dict[i] = np.inf
            for j in self.medoids:
                if i != j:
                    temp_Delta = self.distance_on_multi_dimensions(
                        i, j, dimension_dict[i]
                    )
                    if temp_Delta < Delta_dict[i]:
                        Delta_dict[i] = temp_Delta
        cluster_dict[-1] = []
        for i in self.medoids:
            remove_list = []
            for x in cluster_dict[i]:
                if (
                    self.distance_on_multi_dimensions(i, x, dimension_dict[i])
                    > Delta_dict[i]
                ):
                    cluster_dict[-1].append(x)
                    remove_list.append(x)
            for x in remove_list:
                cluster_dict[i].remove(x)

        self.cluster_dict = cluster_dict
        self.Di = dimension_dict
        making_list = []
        for cluster_id in cluster_dict:
            for index in cluster_dict[cluster_id]:
                making_list.append(
                    {"product_index": index, "cluster_id": cluster_id}
                )
        cluster_df = pd.DataFrame(making_list)
        self.cluster_df = cluster_df
        print("end refinement")
