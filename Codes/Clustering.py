from PCA_and_ETC import *
from sklearn.cluster import *
from scipy.spatial import distance
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

import math
import random


class Clustering:
    def __init__(self, data: pd.DataFrame):
        self.PCA_Data = data
        self.index = data.index

        self.K_Mean = []
        self.DBSCAN = []
        self.Agglomerative = []

        self.bisecting_K_mean = []
        self.HDBSCAN = []
        self.BIRCH = []

        self.OPTIC = []
        self.Gaussian = []
        self.meanshift = []

        self.K_Mean_labels = []
        self.DBSCAN_labels = []
        self.Agglomerative_labels = []

        self.bisecting_K_mean_labels = []
        self.HDBSCAN_labels = []
        self.BIRCH_labels = []

        self.OPTIC_labels = []
        self.Gaussian_labels = []
        self.meanshift_labels = []

    def perform_kmeans(self, k_value: int, alpha: float):
        """
        :param k_value: Number of Clusters
        :param alpha: outlier threshold
        :return: clustering Individual_Result
        """
        self.PCA_Data = pd.DataFrame(self.PCA_Data)
        self.PCA_Data = self.PCA_Data.values[:, 1:].astype(float)

        # sample 갯수가 K보다 작은 경우 k_value = n_sample
        n_sample = self.PCA_Data.shape[0]
        if n_sample <= k_value:
            k_value = n_sample

        # Clustering
        kmeans = KMeans(init='k-means++', n_clusters=k_value, n_init=10, max_iter=500,
                        random_state=42).fit(self.PCA_Data)
        cluster_labels = kmeans.labels_
        self.K_Mean_labels = cluster_labels

        # Outlier Detection
        distance_to_own_centroid = [distance.euclidean(self.PCA_Data[i], kmeans.cluster_centers_[cluster_labels[i]])
                                    for i in range(len(self.PCA_Data))]

        nbrs = NearestNeighbors(n_neighbors=2, p=2).fit(self.PCA_Data)
        distances, indices = nbrs.kneighbors(self.PCA_Data)
        nearest_neighbor_distances = distances[:, 1]

        sorted_nearest_neighbor_distances = sorted(nearest_neighbor_distances)
        epsilon = np.percentile(sorted_nearest_neighbor_distances, alpha * 100)
        outliers = [i for i, dist in enumerate(distance_to_own_centroid) if dist > epsilon]

        clusters_indices = [[] for _ in range(k_value)]
        for i, label in enumerate(cluster_labels):
            if i in outliers:
                continue
            clusters_indices[label].append(i)

        clusters_indices.insert(0, list(outliers))

        # number index를 firm name으로 바꾸어 2차원 리스트로 저장.
        final_cluster = [[] for _ in clusters_indices]
        for i, num in enumerate(clusters_indices):
            for j in num:
                final_cluster[i].append(self.index[j])

        final_cluster = [cluster for cluster in final_cluster if cluster]
        # final_cluster.insert(0, [])
        self.K_Mean = final_cluster

    def perform_DBSCAN(self, threshold: float):
        """
        :param threshold: outlier threshold
        :return: clustering Individual_Result
        """
        self.PCA_Data = pd.DataFrame(self.PCA_Data)
        self.PCA_Data = self.PCA_Data.values[:, 1:].astype(float)

        # 각 데이터 포인트의 MinPts 개수의 최근접 이웃들의 거리의 평균 계산
        # 1번째는 자기자신이니까 ms+1
        ms = int(np.log(len(self.PCA_Data)))
        # ms=6
        nbrs = NearestNeighbors(n_neighbors=ms + 1, p=1).fit(self.PCA_Data)
        distances, indices = nbrs.kneighbors(self.PCA_Data)
        avg_distances = np.mean(distances[:, 1:], axis=1)
        eps = np.percentile(avg_distances, threshold * 100)

        # Clustering
        dbscan = DBSCAN(min_samples=ms, eps=eps, metric='manhattan').fit(self.PCA_Data)
        cluster_labels = dbscan.labels_
        self.DBSCAN_labels = cluster_labels

        # Get the unique cluster labels
        unique_labels = sorted(list(set(cluster_labels)))

        clust = [[] for _ in unique_labels]
        for i, cluster_label in enumerate(cluster_labels):
            clust[unique_labels.index(cluster_label)].append(self.index[i])

        self.DBSCAN = clust

    def perform_HA(self, threshold: float):
        """
        :param threshold: outlier threshold
        :return: clustering Individual_Result
        """
        self.PCA_Data = pd.DataFrame(self.PCA_Data)
        self.PCA_Data = self.PCA_Data.values[:, 1:].astype(float)

        # 이웃한 두 개 점 사이 거리 alpha percentile
        nbrs = NearestNeighbors(n_neighbors=2, p=2).fit(self.PCA_Data)
        distances, indices = nbrs.kneighbors(self.PCA_Data)
        avg_distances = distances[:, 1:]
        outlier_distance = np.percentile(avg_distances, threshold * 100)

        # Clustering
        agglo = AgglomerativeClustering(n_clusters=None, metric='euclidean', linkage='ward',
                                        distance_threshold=outlier_distance).fit(self.PCA_Data)
        cluster_labels = agglo.labels_
        self.Agglomerative_labels = cluster_labels

        # Outlier Detection
        outlier = []
        for i, avg_distance in enumerate(avg_distances):
            if avg_distance > outlier_distance:
                outlier.append(i)

        for i, cluster_label in enumerate(cluster_labels):
            if i in outlier:
                cluster_labels[i] = -1

        unique_labels = sorted(list(set(cluster_labels)))

        clust = [[] for _ in unique_labels]
        for i, cluster_label in enumerate(cluster_labels):
            clust[unique_labels.index(cluster_label)].append(self.index[i])

        # outlier가 없으면 빈리스트 추가
        if -1 not in unique_labels:
            clust.insert(0, [])

        self.Agglomerative = clust

    def perform_bisectingkmeans(self, k_value: int, alpha: float = 0.5):
        """
        :param k_value: Number of Cluster
        :param alpha: Outlier threshold
        :return: clustering Individual_Result
        """
        self.PCA_Data = pd.DataFrame(self.PCA_Data)
        self.PCA_Data = self.PCA_Data.values[:, 1:].astype(float)

        # sample 갯수가 K보다 작은 경우 k_value = n_sample
        n_sample = self.PCA_Data.shape[0]
        if n_sample <= k_value:
            k_value = n_sample

        # Clustering
        kmeans = MiniBatchKMeans(init='k-means++', n_clusters=k_value, n_init=10, max_iter=500,
                                 random_state=random.randint(1, 100)).fit(self.PCA_Data)
        cluster_labels = kmeans.labels_
        self.bisecting_K_mean_labels = cluster_labels

        # Outlier Detection
        distance_to_own_centroid = [distance.euclidean(self.PCA_Data[i], kmeans.cluster_centers_[cluster_labels[i]])
                                    for i in range(len(self.PCA_Data))]

        nbrs = NearestNeighbors(n_neighbors=3, p=2).fit(self.PCA_Data)
        distances, indices = nbrs.kneighbors(self.PCA_Data)
        nearest_neighbor_distances = distances[:, 1]

        sorted_nearest_neighbor_distances = sorted(nearest_neighbor_distances)
        epsilon = sorted_nearest_neighbor_distances[int(len(sorted_nearest_neighbor_distances) * alpha)]
        outliers = [i for i, dist in enumerate(distance_to_own_centroid) if dist < epsilon]

        clusters_indices = [[] for _ in range(k_value)]
        for i, label in enumerate(cluster_labels):
            if i in outliers:
                continue
            clusters_indices[label].append(i)
        clusters_indices.insert(0, list(outliers))

        # number index를 firm name으로 바꾸어 2차원 리스트로 저장.
        final_cluster = [[] for _ in clusters_indices]
        for i, num in enumerate(clusters_indices):
            for j in num:
                final_cluster[i].append(self.index[j])

        final_cluster = [cluster for cluster in final_cluster if cluster]
        self.bisecting_K_mean = final_cluster

    def perform_HDBSCAN(self, threshold):
        """
        :param threshold: Outlier threshold
        :return: clustering Individual_Result
        """
        self.PCA_Data = pd.DataFrame(self.PCA_Data)
        self.PCA_Data = self.PCA_Data.values[:, 1:].astype(float)

        # 각 데이터 포인트의 MinPts 개수의 최근접 이웃들의 거리의 평균 계산
        ms = int(math.log(len(self.PCA_Data)))
        nbrs = NearestNeighbors(n_neighbors=ms + 1, p=1).fit(self.PCA_Data)
        distances, indices = nbrs.kneighbors(self.PCA_Data)
        avg_distances = np.max(distances[:, 1:], axis=1)
        eps = np.percentile(avg_distances, threshold * 100)

        # Clustering
        Hdbscan = HDBSCAN(min_cluster_size=ms, cluster_selection_epsilon=eps, allow_single_cluster=False).fit(
            self.PCA_Data)
        cluster_labels = Hdbscan.labels_
        self.HDBSCAN_labels = cluster_labels

        # Get the unique cluster labels
        unique_labels = sorted(list(set(cluster_labels)))

        clust = [[] for _ in unique_labels]
        for i, cluster_label in enumerate(cluster_labels):
            clust[unique_labels.index(cluster_label)].append(self.index[i])

        # outlier가 없으면 빈리스트 추가
        if -1 not in unique_labels:
            clust.insert(0, [])

        self.HDBSCAN = clust

    def perform_BIRCH(self, threshold):
        """
        :param threshold: Outlier threshold
        :return: clustering Individual_Result
        """
        self.PCA_Data = pd.DataFrame(self.PCA_Data)
        self.PCA_Data = self.PCA_Data.values[:, 1:].astype(float)

        # 이웃한 두 개 점 사이 거리의 평균 계산
        nbrs = NearestNeighbors(n_neighbors=3, p=1).fit(self.PCA_Data)
        distances, indices = nbrs.kneighbors(self.PCA_Data)
        avg_distances = np.max(distances[:, 1:], axis=1)
        outlier_distance = np.percentile(avg_distances, threshold * 100)

        # Clustering
        birch = Birch(threshold=outlier_distance, n_clusters=None).fit(self.PCA_Data)
        cluster_labels = birch.labels_
        self.BIRCH_labels = cluster_labels

        # Outlier Detection
        outlier = []
        for i, avg_distance in enumerate(avg_distances):
            if avg_distance > outlier_distance:
                outlier.append(i)

        for i, cluster_label in enumerate(cluster_labels):
            if i in outlier:
                cluster_labels[i] = -1

        unique_labels = sorted(list(set(cluster_labels)))

        clust = [[] for _ in unique_labels]
        for i, cluster_label in enumerate(cluster_labels):
            clust[unique_labels.index(cluster_label)].append(self.index[i])

        # outlier가 없으면 빈리스트 추가
        if -1 not in unique_labels:
            clust.insert(0, [])

        self.BIRCH = clust

    def perform_OPTICS(self, threshold):
        """
        :param threshold: outlier threshold
        :return: clustering Individual_Result
        """
        self.PCA_Data = pd.DataFrame(self.PCA_Data)
        self.PCA_Data = self.PCA_Data.values[:, 1:].astype(float)

        # Clustering
        optics = OPTICS(cluster_method='xi', xi=threshold, min_cluster_size=0.1, metric='manhattan').fit(
            self.PCA_Data)
        labels = optics.labels_
        self.OPTIC_labels = labels

        unique_labels = sorted(list(set(labels)))

        clust = [[] for _ in unique_labels]
        for i, cluster_label in enumerate(labels):
            clust[unique_labels.index(cluster_label)].append(self.index[i])

        # outlier가 없으면 빈리스트 추가
        if -1 not in unique_labels:
            clust.insert(0, [])

        self.OPTIC = clust

    def perform_meanshift(self, quantile):
        """
        :param quantile: outlier threshold
        :return: clustering Individual_Result
        """
        self.PCA_Data = pd.DataFrame(self.PCA_Data)
        self.PCA_Data = self.PCA_Data.values[:, 1:].astype(float)

        # The following bandwidth can be automatically detected using
        bandwidth = estimate_bandwidth(self.PCA_Data, quantile=quantile)
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, cluster_all=False).fit(self.PCA_Data)
        cluster_labels = ms.labels_
        self.meanshift_labels = cluster_labels

        # Get the unique cluster labels
        unique_labels = sorted(list(set(self.meanshift_labels)))

        clusters = [[] for _ in unique_labels]
        for i, cluster_label in enumerate(self.meanshift_labels):
            clusters[unique_labels.index(cluster_label)].append(self.index[i])

        # outlier가 없으면 빈리스트 추가
        if -1 not in unique_labels:
            clusters.insert(0, [])

        self.meanshift = clusters

    def perform_GMM(self, n_components: float):
        """
        :param n_components: Number of clusters
        :return: clustering Individual_Result
        """
        self.PCA_Data = pd.DataFrame(self.PCA_Data)
        self.PCA_Data = self.PCA_Data.values[:, 1:].astype(float)

        # 1. Gaussian Mixture Model
        gmm = GaussianMixture(n_components=n_components, init_params='k-means++', covariance_type='full').fit(
            self.PCA_Data)
        cluster_labels = gmm.predict(self.PCA_Data)
        self.Gaussian_labels = cluster_labels

        clusters = [[] for _ in range(n_components)]

        for i, cluster_num in enumerate(cluster_labels):
            clusters[cluster_num].append(i)

        clusters = [sublist for sublist in clusters if sublist]

        # Outliers
        probabilities = gmm.score_samples(self.PCA_Data)
        threshold = np.percentile(probabilities, 70)

        outliers = []
        for i, probability in enumerate(probabilities):
            if probability < threshold:
                outliers.append(i)

        # a에 있는 값을 b에서 빼기
        for value in outliers:
            for i, row in enumerate(clusters):
                if value in row:
                    clusters[i].remove(value)

        # 1차원 리스트로 전환된 outlier를 cluster 맨앞에 저장.
        clusters.insert(0, outliers)

        for i, cluster in enumerate(clusters):
            for t, num in enumerate(cluster):
                cluster[t] = self.index[num]

        self.Gaussian = clusters
