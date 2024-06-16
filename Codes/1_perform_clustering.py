from PCA_and_ETC import *

import Clustering as C

input_dir = '../Database/characteristics_US'
output_dir = '../Database/Clustering_Result'

# hyper parameter K(3, 5, 10, 25, 50, 75, 100, 200, 300) should be tested manually.(paper follow)
K_mean_Save = False
if K_mean_Save:
    test_Set = [i for i in [10, 20, 30, 50, 100]]
    files = sorted(filename for filename in os.listdir(input_dir))

    for i in test_Set:
        for file in files:
            print(file)
            data = read_and_preprocess_data(input_dir, file)

            fundamental = True
            if not fundamental:
                a = momentum_prefix_finder(data)
                columns_to_drop = [col for col in data.columns if a not in col]
                data = data.drop(columns=columns_to_drop)

            PCA = True
            if PCA:
                df_combined = generate_PCA_Data(data)
            if not PCA:
                df_combined = data

            Do_Clustering = C.Clustering(df_combined)
            Do_Clustering.perform_kmeans(i, 0.5)

            Result = pd.DataFrame()
            Result.index = df_combined.index
            Result['mom1'] = df_combined.iloc[:, 0]
            for j in range(len(Do_Clustering.K_Mean)):
                Result.loc[Result.index.isin(Do_Clustering.K_Mean[j]), 'clusters'] = j
            print("Number of clusters is:", len(set(Do_Clustering.K_Mean_labels)))

            sub_dir = f'K_mean/{i}/{file}'
            Result.reset_index(inplace=True)
            Result['clusters'] = Result['clusters'].astype(int)
            Result.to_csv(os.path.join(output_dir, sub_dir), index=False)

# hyper parameter eps percentile np.range(0.1, 1, 0.1) should be tested manually.(paper follow)
dbscan_Save = True
if dbscan_Save:
    test_Set = [round(i, 1) for i in np.arange(0.1, 1, 0.1)]
    files = sorted(filename for filename in os.listdir(input_dir))

    for i in test_Set:
        for file in files:
            print(file)
            data = read_and_preprocess_data(input_dir, file)

            fundamental = True
            if not fundamental:
                a = momentum_prefix_finder(data)
                columns_to_drop = [col for col in data.columns if a not in col]
                data = data.drop(columns=columns_to_drop)

            PCA = True
            if PCA:
                df_combined = generate_PCA_Data(data)
            if not PCA:
                df_combined = data

            Do_Clustering = C.Clustering(df_combined)
            Do_Clustering.perform_DBSCAN(i)

            Result = pd.DataFrame()
            Result.index = df_combined.index
            Result['mom1'] = df_combined.iloc[:, 0]
            for j in range(len(Do_Clustering.DBSCAN)):
                Result.loc[Result.index.isin(Do_Clustering.DBSCAN[j]), 'clusters'] = j
            print("Number of clusters is:", len(set(Do_Clustering.DBSCAN_labels)))

            sub_dir = f'DBSCAN/{i}/{file}'
            Result.reset_index(inplace=True)
            Result['clusters'] = Result['clusters'].astype(int)
            Result.to_csv(os.path.join(output_dir, sub_dir), index=False)

# hyper parameter distance percentile np.range(0.1, 1, 0.1) should be tested manually.(paper follow)
agglomerative_Save = False
if agglomerative_Save:
    test_Set = [round(i, 1) for i in np.arange(0.1, 1, 0.1)]
    files = sorted(filename for filename in os.listdir(input_dir))

    for i in test_Set:
        for file in files:
            print(file)
            data = read_and_preprocess_data(input_dir, file)

            fundamental = True
            if not fundamental:
                a = momentum_prefix_finder(data)
                columns_to_drop = [col for col in data.columns if a not in col]
                data = data.drop(columns=columns_to_drop)

            PCA = True
            if PCA:
                df_combined = generate_PCA_Data(data)
            if not PCA:
                df_combined = data

            Do_Clustering = C.Clustering(df_combined)
            Do_Clustering.perform_HA(i)

            Result = pd.DataFrame()
            Result.index = df_combined.index
            Result['mom1'] = df_combined.iloc[:, 0]
            for j in range(len(Do_Clustering.Agglomerative)):
                Result.loc[Result.index.isin(Do_Clustering.Agglomerative[j]), 'clusters'] = j
            print("Number of clusters is:", len(set(Do_Clustering.Agglomerative_labels)))

            sub_dir = f'Agglomerative/{i}/{file}'
            Result.reset_index(inplace=True)
            Result['clusters'] = Result['clusters'].astype(int)
            Result.to_csv(os.path.join(output_dir, sub_dir), index=False)
