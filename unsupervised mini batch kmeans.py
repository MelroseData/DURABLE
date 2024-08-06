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

from sklearn.preprocessing import MinMaxScaler
class MyModel:
    """
    that's more simply, I just use a linear model within those analysis to make it link with biology rules
    about the validation :
    replace NaNs and infs with 1, and I add epsilon before taking log
    Ensure all p-values are > 0 and <= 1, after calculation, print log pvalues before combine_pvalue
    I do them in the end with 10 clusters
    """

    def __init__(self, data_paths, analysis_results=None):
        self.data_paths = data_paths
        self.analysis_results = analysis_results if analysis_results is not None else []

    def load_and_preprocess_data(self, path):
        """
        Loading the Data into a Dask DataFrame and using this function:
        dask_df = dd.read_csv(path, assume_missing=True, sample=1000000)
        doesn't mean I only take 1000000 samples but a way to handle the reading process
        Then use the 'Unnamed: 0' column as the index of the DataFrame
        Converting to a Pandas DataFrame to match the later calculation
        Clear all the missing value and converting Back to a Dask DataFrame then to a Sparse Format
        Dropping columns that are the same as the first row and returning the DataFrame
        otherwise your output will be red
        """
        print(f"Loading and preprocessing data from {path}")
        dask_df = dd.read_csv(path, assume_missing=True, sample=1000000)
        dask_df = dask_df.set_index('Unnamed: 0')
        print(dask_df.index)
        pandas_df = dask_df.compute()
        pandas_df = pandas_df.dropna(how='all', axis=1)
        scaler = MinMaxScaler()
        normalized_data = pd.DataFrame(scaler.fit_transform(pandas_df), columns=pandas_df.columns)
        dask_df = dd.from_pandas(normalized_data, npartitions=5)
        dask_df = dask_df.astype(pd.SparseDtype("float", np.nan))
        first_row = dask_df.head(1)
        dask_df = dask_df.loc[:, (dask_df.head(1) != first_row).any()]
        return dask_df

    def correlation_analysis(self, data):
        print("Performing correlation analysis")
        correlations = data.corr()
        self.analysis_results.append({'type': 'correlation', 'data': correlations})
        print(correlations)
        return correlations

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
        print(data1.index)
        print(data2.index)
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

    def unsupervised_learning(self, data):

        correlation_results = [result['data'] for result in self.analysis_results if
                               result['type'] == 'correlation' and result['data'] is not None]

        causal_results = [result['data'] for result in self.analysis_results if
                          result['type'] == 'causal' and result['data'] is not None]
        print(data)
        print(data.index)
        if correlation_results:
            non_zero_correlations = correlations.replace(0, np.nan).dropna(axis=1, how='any')
            non_zero_cell_lines = non_zero_correlations.index
            print(non_zero_correlations)
            print(non_zero_cell_lines)
            if not non_zero_cell_lines.empty:
                existing_indices = non_zero_cell_lines.intersection(data.index)
                print(existing_indices)
                if not existing_indices.empty:
                    data = data.loc[existing_indices]
                else:
                    print("No matching indices found in the DataFrame.")
            else:
                print("No correlations found.")

        if causal_results:
            strong_causal_effects = [i for i, result in enumerate(causal_results) if result != 0]
            print(strong_causal_effects)
            if strong_causal_effects:
                existing_indices = set(strong_causal_effects).intersection(set(data.index))
                print(existing_indices)
                if existing_indices:
                    data = data.loc[list(existing_indices)]
                else:
                    print("No matching indices found in the DataFrame.")
            else:
                print("No causal effects found.")
        print(f"Performing unsupervised learning on dataset with shape {data.shape}")
        kmeans = MiniBatchKMeans(n_clusters=3)
        kmeans.fit(data)
        labels = kmeans.labels_
        score = silhouette_score(data, labels)
        print(f'Silhouette score: {score}')
        return labels


if __name__ == "__main__":

    FatherPath = Path(__file__).parents[1]
    CurrentPath = FatherPath / "unsupervised pro"
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
    Level4_dataset = my_model.load_and_preprocess_data(Level4Path).compute()
    Level4_dataset = Level4_dataset.astype(pd.SparseDtype("float", 0))
    print("**************************************************************************")
    labels = my_model.unsupervised_learning(Level4_dataset)
    gc.collect()
