from utils.PCA_and_ETC import *
from sklearn.cluster import *
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors


class Clustering:
    def __init__(self, data: pd.DataFrame):
        self.PCA_Data = data
        self.index = data.index

        self.K_Mean = []
        self.DBSCAN = []
        self.Agglomerative = []

        self.K_Mean_labels = []
        self.DBSCAN_labels = []
        self.Agglomerative_labels = []

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
        agglo = AgglomerativeClustering(n_clusters=None, metric='euclidean', linkage='average',
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
