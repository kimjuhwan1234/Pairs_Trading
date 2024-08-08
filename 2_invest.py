from utils.PCA_and_ETC import *

from Module import Cointegration as CI, Invest_Table as I

# hyper parameter K(3, 5, 10, 25, 50, 75, 100, 200, 300) should be tested manually.(paper follow)
K_mean_Save = False
if K_mean_Save:
    input_dir = 'Database/Clustering_Result/K_mean'
    subdirectories = [d for d in os.listdir(input_dir)]

    for subdir in subdirectories:
        top_df = pd.DataFrame(columns=['month', 'invested', 'outlier', 'first', 'second',
                                       'rest', 'total', 'number of clusters'])
        base_directory = f'{input_dir}/{subdir}'
        output_dir = f'../Database/LS_Result/K_mean/{subdir}'
        files = sorted(filename for filename in os.listdir(base_directory) if filename.endswith('.csv'))

        for file in files:
            print(file)
            data = read_and_preprocess_data(base_directory, file)

            clusters = []
            for j in range(1 + max(set(data['clusters']))):
                indices = list(data[data['clusters'] == j].index)
                clusters.append(indices)

            Do_Result_Save = I.Invest_Table(data)
            Do_Result_Save.ls_table(clusters, output_dir, file, save=True, raw=True)

            sublist_lengths = [len(sublist) for sublist in clusters]
            outlier_lengths = sublist_lengths[0]
            cluster_lengths = sublist_lengths[1:]
            top2_lengths = sorted(set(cluster_lengths), reverse=True)[:2]
            rest = np.sum(cluster_lengths) - np.sum(top2_lengths)

            if len(top2_lengths) == 1:
                top2_lengths.append(0)

            if len(top2_lengths) == 0:
                top2_lengths.append(0)
                top2_lengths.append(0)

            new_row = pd.DataFrame({'month': [file[:-4]],
                                    'invested': Do_Result_Save.count_stock_of_traded(),
                                    'outlier': outlier_lengths,
                                    'first': top2_lengths[0],
                                    'second': top2_lengths[1],
                                    'rest': rest,
                                    'total': len(data),
                                    'number of clusters': len(sublist_lengths[1:])})

            top_df = pd.concat([top_df, new_row], ignore_index=True)
        top_df.to_csv(os.path.join('File/Individual_Result/K_mean', f'{subdir}.csv'), index=False)

# hyper parameter eps percentile np.range(0.1, 1, 0.1) should be tested manually.(paper follow) Done!
dbscan_Save = False
if dbscan_Save:
    input_dir = 'Database/Clustering_Result/DBSCAN'
    subdirectories = [d for d in os.listdir(input_dir)]

    for subdir in subdirectories:
        top_df = pd.DataFrame(columns=['month', 'invested', 'outlier', 'first', 'second',
                                       'rest', 'total', 'number of clusters'])
        base_directory = f'{input_dir}/{subdir}'
        output_dir = f'../Database/LS_Result/DBSCAN/{subdir}'
        files = sorted(filename for filename in os.listdir(base_directory) if filename.endswith('.csv'))

        for file in files:
            print(file)
            data = read_and_preprocess_data(base_directory, file)

            clusters = []
            for j in range(1 + max(set(data['clusters']))):
                indices = list(data[data['clusters'] == j].index)
                clusters.append(indices)

            Do_Result_Save = I.Invest_Table(data)
            Do_Result_Save.ls_table(clusters, output_dir, file, save=True, raw=True)

            sublist_lengths = [len(sublist) for sublist in clusters]
            outlier_lengths = sublist_lengths[0]
            cluster_lengths = sublist_lengths[1:]
            top2_lengths = sorted(set(cluster_lengths), reverse=True)[:2]
            rest = np.sum(cluster_lengths) - np.sum(top2_lengths)

            if len(top2_lengths) == 1:
                top2_lengths.append(0)

            if len(top2_lengths) == 0:
                top2_lengths.append(0)
                top2_lengths.append(0)

            new_row = pd.DataFrame({'month': [file[:-4]],
                                    'invested': Do_Result_Save.count_stock_of_traded(),
                                    'outlier': outlier_lengths,
                                    'first': top2_lengths[0],
                                    'second': top2_lengths[1],
                                    'rest': rest,
                                    'total': len(data),
                                    'number of clusters': len(sublist_lengths[1:])})

            top_df = pd.concat([top_df, new_row], ignore_index=True)
        top_df.to_csv(os.path.join('File/Individual_Result/DBSCAN', f'{subdir}.csv'), index=False)

# hyper parameter distance percentile np.range(0.1, 1, 0.1) should be tested manually.(paper follow) Done!
agglomerative_Save = False
if agglomerative_Save:
    input_dir = 'Database/Clustering_Result/Agglomerative'
    subdirectories = [d for d in os.listdir(input_dir)]

    for subdir in subdirectories:
        top_df = pd.DataFrame(columns=['month', 'invested', 'outlier', 'first', 'second',
                                       'rest', 'total', 'number of clusters'])
        base_directory = f'{input_dir}/{subdir}'
        output_dir = f'../Database/LS_Result/Agglomerative/{subdir}'
        files = sorted(filename for filename in os.listdir(base_directory) if filename.endswith('.csv'))

        for file in files:
            print(file)
            data = read_and_preprocess_data(base_directory, file)

            clusters = []
            for j in range(1 + max(set(data['clusters']))):
                indices = list(data[data['clusters'] == j].index)
                clusters.append(indices)

            Do_Result_Save = I.Invest_Table(data)
            Do_Result_Save.ls_table(clusters, output_dir, file, save=True, raw=True)

            sublist_lengths = [len(sublist) for sublist in clusters]
            outlier_lengths = sublist_lengths[0]
            cluster_lengths = sublist_lengths[1:]
            top2_lengths = sorted(set(cluster_lengths), reverse=True)[:2]
            rest = np.sum(cluster_lengths) - np.sum(top2_lengths)

            if len(top2_lengths) == 1:
                top2_lengths.append(0)

            if len(top2_lengths) == 0:
                top2_lengths.append(0)
                top2_lengths.append(0)

            new_row = pd.DataFrame({'month': [file[:-4]],
                                    'invested': Do_Result_Save.count_stock_of_traded(),
                                    'outlier': outlier_lengths,
                                    'first': top2_lengths[0],
                                    'second': top2_lengths[1],
                                    'rest': rest,
                                    'total': len(data),
                                    'number of clusters': len(sublist_lengths[1:])})

            top_df = pd.concat([top_df, new_row], ignore_index=True)
        top_df.to_csv(os.path.join('File/Individual_Result/Agglomerative', f'{subdir}.csv'), index=False)

contrastive = False
if contrastive:
    input_dir = f'Database/Clustering_Result/Contrastive_Learning2/CL_pre_0.8'
    subdirectories = [d for d in os.listdir(input_dir)]

    for subdir in subdirectories:
        top_df = pd.DataFrame(columns=['month', 'invested', 'outlier', 'first', 'second',
                                       'rest', 'total', 'number of clusters'])
        base_directory = f'{input_dir}/{subdir}'
        output_dir = f'../Database/LS_Result/Contrastive_Learning2/{subdir}'
        files = sorted(filename for filename in os.listdir(base_directory) if filename.endswith('.csv'))

        for file in files:
            print(file)
            data = read_and_preprocess_data(base_directory, file)

            clusters = []
            for j in range(1 + max(set(data['clusters']))):
                indices = list(data[data['clusters'] == j].index)
                clusters.append(indices)

            Do_Result_Save = I.Invest_Table(data)
            Do_Result_Save.ls_table(clusters, output_dir, file, save=True, raw=True)

            sublist_lengths = [len(sublist) for sublist in clusters]
            outlier_lengths = sublist_lengths[0]
            cluster_lengths = sublist_lengths[1:]
            top2_lengths = sorted(set(cluster_lengths), reverse=True)[:2]
            rest = np.sum(cluster_lengths) - np.sum(top2_lengths)

            if len(top2_lengths) == 1:
                top2_lengths.append(0)

            if len(top2_lengths) == 0:
                top2_lengths.append(0)
                top2_lengths.append(0)

            new_row = pd.DataFrame({'month': [file[:-4]],
                                    'invested': Do_Result_Save.count_stock_of_traded(),
                                    'outlier': outlier_lengths,
                                    'first': top2_lengths[0],
                                    'second': top2_lengths[1],
                                    'rest': rest,
                                    'total': len(data),
                                    'number of clusters': len(sublist_lengths[1:])})

            top_df = pd.concat([top_df, new_row], ignore_index=True)
        top_df.to_csv(os.path.join(f'File/Individual_Result/Contrastive_Learning2', f'{subdir}.csv'), index=False)

Reversal_Save = False
if Reversal_Save:
    input_dir = 'Database/characteristics_US'
    output_dir = 'Database/LS_Result/Reversal'
    files = sorted(filename for filename in os.listdir(input_dir))

    for file in files:
        print(file)

        if file in os.listdir(output_dir):
            continue

        data = read_and_preprocess_data(input_dir, file)
        Do_Result_Save = I.Invest_Table(data)
        Do_Result_Save.reversal_table(data, output_dir, file)

cointegration = False
if cointegration:
    input_dir = 'Database/characteristics_US'
    output_dir = '../Database/LS_Result/Total/Cointegration'
    files = sorted(filename for filename in os.listdir(input_dir))

    for file in files:
        print(file)

        if file in os.listdir(output_dir):
            continue

        # if int(file[:4]) < 2021:
        #     continue

        data = read_and_preprocess_data(input_dir, file)
        Coin = CI.cointegration(output_dir, file, data)
        Coin.read_mom_data()
        Coin.find_cointegrated_pairs()
        Coin.save_cointegrated_LS()

Best = True
if Best:
    input_dir = f'Database/Clustering_Result/Best'
    subdirectories = [d for d in os.listdir(input_dir)]

    for subdir in subdirectories:
        top_df = pd.DataFrame(columns=['month', 'invested', 'outlier', 'first', 'second',
                                       'rest', 'total', 'number of clusters'])
        base_directory = f'{input_dir}/{subdir}'
        output_dir = f'../Database/LS_Result/Best2/{subdir}'
        files = sorted(filename for filename in os.listdir(base_directory) if filename.endswith('.csv'))

        for file in files:
            print(file)
            data = read_and_preprocess_data(base_directory, file)

            clusters = []
            for j in range(1 + max(set(data['clusters']))):
                indices = list(data[data['clusters'] == j].index)
                clusters.append(indices)

            Do_Result_Save = I.Invest_Table(data)
            Do_Result_Save.ls_table(clusters, output_dir, file, save=True, raw=True)
