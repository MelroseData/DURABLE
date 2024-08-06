from sklearn.cluster import MiniBatchKMeans
from pathlib import Path
from sklearn.linear_model import SGDRegressor
from multiprocessing import Pool
import warnings
import numpy as np
import gc
import dask.dataframe as dd
from scipy.stats import combine_pvalues
import pandas as pd
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score


class MyModel:


    def __init__(self, data_paths, analysis_results=None):
        self.data_paths = data_paths
        self.analysis_results = analysis_results if analysis_results is not None else []

    def load_and_preprocess_data(self, path):
        print(f"Loading and preprocessing data from {path}")
        dask_df = dd.read_csv(path, assume_missing=True, sample=1000000)
        dask_df = dask_df.set_index('Unnamed: 0')
        pandas_df = dask_df.compute()
        pandas_df = pandas_df.dropna(how='all', axis=1)
        dask_df = dd.from_pandas(pandas_df, npartitions=5)
        dask_df = dask_df.astype(pd.SparseDtype("float", np.nan))
        first_row = dask_df.head(1)
        dask_df = dask_df.loc[:, (dask_df.head(1) != first_row).any()]
        return dask_df

    def correlation_analysis(self, data):
        print("Performing correlation analysis")
        correlations = data.corr().values.flatten().tolist()
        self.analysis_results.append({'type': 'correlation', 'data': correlations})
        return {'type': 'correlation', 'data': correlations}

    @staticmethod
    def estimate_causal_effect(args):
        data1, data2, outcome1, outcome2 = args
        X = data1[outcome1]
        y = data2[outcome2]
        y = y.loc[X.index]
        y = y.fillna(0)
        X = X.fillna(X.mean())
        model = SGDRegressor().fit(X.values.to_dense().reshape(-1, 1), y)
        coef = model.coef_[0]
        return {'type': 'causal', 'data': coef}

    def estimate_causal_effects(self, data_path1, data_path2):
        print("Estimating causal effects")
        data1 = self.load_and_preprocess_data(data_path1).compute()
        data2 = self.load_and_preprocess_data(data_path2).compute()
        data1 = data1.dropna(how='all')
        data2 = data2.dropna(how='all')
        common_index = data1.index.intersection(data2.index)
        data1 = data1.loc[common_index]
        data2 = data2.loc[common_index]
        outcomes1 = data1.columns.tolist()
        outcomes2 = data2.columns.tolist()
        def args_generator():
            for outcome1 in outcomes1:
                for outcome2 in outcomes2:
                    yield (data1, data2, outcome1, outcome2)
        with Pool(5) as p:
            self.analysis_results.extend(p.map(MyModel.estimate_causal_effect, args_generator()))
        p.close()
        p.join()
        gc.collect()

    def combine_analysis_results(self):
        print("Combining analysis results")
        combined_results = []
        for result in self.analysis_results:
            if result['type'] == 'correlation' and result['data']:
                pvalues = np.array(result['data'])
                print(f'Initial pvalues: {pvalues}')
                pvalues = np.where(np.isnan(pvalues) | np.isinf(pvalues), 1, pvalues)
                print(
                    f'Pvalues after replacing NaNs and infs: {pvalues}')
                pvalues = np.where(pvalues <= 0, 1e-3, pvalues)
                pvalues = np.where(pvalues > 1, 1, pvalues)
                print(
                    f'Pvalues after replacing zeros and ensuring <= 1: {pvalues}')
                print(f'Min pvalue before log: {np.min(pvalues)}')
                epsilon = 1e-6
                log_pvalues = np.log1p(pvalues + epsilon)
                print(f'Log pvalues: {log_pvalues}')
                print(f'Log pvalues before combine_pvalues: {log_pvalues}')
                print(f'Log pvalues before sum: {log_pvalues}')
                statistic = -2 * np.sum(log_pvalues)
                print(f'Statistic: {statistic}')
                combined_statistic, combined_pvalue = combine_pvalues(log_pvalues)
                combined_results.append(
                    {'type': 'correlation', 'statistic': combined_statistic, 'pvalue': combined_pvalue,
                     'sum_log_pvalues': statistic})
            elif result['type'] == 'causal':
                effect_size = calculate_effect_size(result['data'])
                variance = calculate_variance(result['data'])
                combined_results.append({'type': 'causal', 'effect_size': effect_size, 'variance': variance})
        return combined_results

    def unsupervised_learning(self, data):
        print(f"Performing unsupervised learning on dataset with shape {data.shape}")
        kmeans = MiniBatchKMeans(n_clusters=3)
        kmeans.fit(data)
        labels = kmeans.labels_
        score = silhouette_score(data, labels)
        print(f'Silhouette score: {score}')
        data_dense = pd.DataFrame(data)
        data_dense = data.values
        pca = PCA(n_components=2)
        data_2d = pca.fit_transform(data_dense)
        plt.scatter(data_2d[:, 0], data_2d[:, 1], c=labels)
        plt.show()
        inertias = []
        num_clusters = range(1, 21)
        for k in num_clusters:
            kmeans = MiniBatchKMeans(n_clusters=k)
            kmeans.fit(data)
            inertias.append(kmeans.inertia_)
        plt.plot(num_clusters, inertias, 'bx-')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Inertia')
        plt.title('The Elbow Method showing the optimal k')
        plt.show()
        data_dense = pd.DataFrame(data)
        data_dense = data_dense.values
        pca = PCA(n_components=3)
        data_3d = pca.fit_transform(data_dense)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data_3d[:, 0], data_3d[:, 1], data_3d[:, 2], c=labels)
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_zlabel('Feature 3')
        plt.show()

        return labels

    @staticmethod
    def stability_analysis_kmeans(data, n_clusters_range=(3, 10), n_runs=10):
        stability_scores = []

        for _ in range(n_runs):
            n_clusters = np.random.randint(*n_clusters_range)

            kmeans = MiniBatchKMeans(n_clusters=n_clusters)
            kmeans.fit(data)
            labels = kmeans.labels_

            if _ == 0:
                original_labels = labels
            else:
                stability_score = adjusted_rand_score(original_labels, labels)
                stability_scores.append(stability_score)

            original_labels = labels

        return np.mean(stability_scores), np.std(stability_scores), n_runs


if __name__ == "__main__":


    FatherPath = Path(__file__).parents[1]
    CurrentPath = FatherPath / "clear"
    Level1Path = str(CurrentPath / "L1dataset.csv")
    Level2Path = str(CurrentPath / "L2dataset.csv")
    Level3Path = str(CurrentPath / "L3dataset.csv")
    Level4Path = str(CurrentPath / "Drug_sensitivity_AUC_(Sanger_GDSC2)_upgrade.csv")
    my_model = MyModel([Level1Path, Level2Path, Level3Path])
    warnings.filterwarnings("ignore", category=FutureWarning)

    for path in my_model.data_paths:
        dataset = my_model.load_and_preprocess_data(path).compute()
        correlations = my_model.correlation_analysis(dataset)
        gc.collect()
    print("**************************************************************************")
    for i in range(len(my_model.data_paths) - 1):
        my_model.estimate_causal_effects(my_model.data_paths[i], my_model.data_paths[i + 1])
    print("**************************************************************************")
    def calculate_effect_size(result):
        return result.mean()
    def calculate_variance(result):
        return result.var()
    my_model.analysis_results = my_model.combine_analysis_results()

    Level4_dataset = my_model.load_and_preprocess_data(Level4Path).compute()
    Level4_dataset = Level4_dataset.astype(pd.SparseDtype("float", 0))
    print("**************************************************************************")
    labels = my_model.unsupervised_learning(Level4_dataset)
    mean_score, std_score, n_runs = my_model.stability_analysis_kmeans(Level4_dataset, n_clusters_range=(3, 10), n_runs=10)

    # Print the results
    print(f"Mean Stability Score: {mean_score}")
    print(f"Standard Deviation of Stability Scores: {std_score}")
    print(f"Number of Runs: {n_runs}")
    gc.collect()

