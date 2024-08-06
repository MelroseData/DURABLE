from torch.utils.data import  DataLoader
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from multiprocessing import Pool
from sklearn.linear_model import SGDRegressor
import pandas as pd
import numpy as np
import dask.dataframe as dd
import gc
import warnings
from pathlib import Path
import torch.nn.functional as F
import dask.array as da
import torch
import torch.nn as nn

""" 
the main part of the algorithm, 
I got two part, one is the linear, 
another is a feed-forward neural network

the attributes:
n_users: the patient's cell line dna slice,
n_items: those anti-cancer drugs,
n_corr_features: I will do that in the console part
n_causal_features: just like above
n_features1:just like above
n_features3:just like above
n_features4:just like above
n_features5:just like above
embed_dim=50 (I set that with a default value, you can change that whatever you like
correlation: the previous result, i gonna do them auto, you don't have to deal with that
causal_effect: same as the above.
"""

class WideAndDeepModel(nn.Module):
    """
        Normalize the correlation results to the range [0, 1]
        because that can't hold the data, so I could only slice them all into 400*400
        in order to run smoothly, I just reshape to [160000, 1]
        the hiddenlayer is the place where I could do my bio-model freely, so I use the result of the analysis to be the weight
        and adjust each layer to match
        and the nn's attributes I just use the default number 50, you can change that freely
        """

    def __init__(self, n_users, n_items, n_corr_features, n_causal_features, n_features1,n_features3,n_features4,n_features5  ,embed_dim=50,
                 correlation=None, causal_effect=None):
        super().__init__()
        self.layer1 = nn.Linear(n_features1, 400)
        self.layer3 = nn.Linear(n_features3, 400)
        self.layer4 = nn.Linear(n_features4, 400)
        self.layer5 = nn.Linear(n_features5, 400)


        if correlation is not None:
            self.layer1.weight.data.fill_(0.3)
            correlation = (correlation - torch.min(correlation)) / (torch.max(correlation) - torch.min(correlation))
            correlation0_reshaped = correlation[0].view(-1, 1)
            self.layer3 = nn.Linear(1, 1000)
            self.layer3.weight.data = self.layer3(correlation0_reshaped)
            correlation1_reshaped = correlation[1].view(-1, 1)
            self.layer5 = nn.Linear(1, 1000)
            self.layer5.weight.data = self.layer5(correlation1_reshaped)


        if causal_effect is not None:
            causal_effect = (causal_effect - torch.min(causal_effect)) / (
                    torch.max(causal_effect) - torch.min(causal_effect))
            causal_effect0_reshaped = causal_effect[0].view(-1, 1)
            causal_effect0_reshaped = causal_effect0_reshaped.float()
            self.layer4 = nn.Linear(1, 1000)
            self.layer4.weight.data = self.layer4(causal_effect0_reshaped)


        self.output_layer = nn.Linear(50, 1)
        self.user_embedding = nn.Embedding(n_users, embed_dim)
        self.item_embedding = nn.Embedding(n_items, embed_dim)
        self.wide = nn.Linear(176 * embed_dim, 1)
        self.deep_corr = nn.Sequential(
            nn.Linear(n_corr_features + 2 * embed_dim, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 1)
        )
        self.deep_causal = nn.Sequential(
            nn.Linear(n_causal_features + 2 * embed_dim, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 1)
        )



    def forward(self, users, items, x1, x3,x4, x5):
        print(f"Shape of users: {users.shape}")
        print(f"Shape of items: {items.shape}")
        print(f"Shape of x1: {x1.shape}")
        print(f"Shape of x3: {x3.shape}")
        print(f"Shape of x4: {x4.shape}")
        print(f"Shape of x5: {x5.shape}")
        user_embed = self.user_embedding(users)
        item_embed = self.item_embedding(items)
        x1 = F.relu(self.layer1(x1))
        x3 = F.relu(self.layer3(x3))
        x4 = F.relu(self.layer4(x4))
        x5 = F.relu(self.layer5(x5))
        wide_inputs = torch.cat([user_embed, item_embed], dim=1)
        wide_inputs = wide_inputs.view(wide_inputs.size(0), -1)
        wide_outputs = self.wide(wide_inputs)

        deep_corr_inputs = torch.cat([x1, x3, x5, user_embed, item_embed], dim=1)
        deep_corr_outputs = self.deep_corr(deep_corr_inputs)

        deep_causal_inputs = torch.cat([x4, user_embed, item_embed], dim=1)
        deep_causal_outputs = self.deep_causal(deep_causal_inputs)

        outputs = wide_outputs + deep_corr_outputs + deep_causal_outputs
        return outputs


class MyModel:

    def __init__(self, data_paths, embedding_files, analysis_results=None):
        self.data_paths = data_paths
        self.embedding_files = embedding_files
        self.analysis_results = analysis_results if analysis_results is not None else []
        self.wide_and_deep_model = None

    def initialize_wide_and_deep_model(self, n_users, n_items, n_features, embed_dim=50):
        self.wide_and_deep_model = WideAndDeepModel(n_users, n_items, n_features, embed_dim)

    def preprocess_data(self, dask_df):
        """
                the beginning of the preprocessing
                the first part, which is the purify and the preprocessing python files
                give the relatively general preprocessing despite the algorithm

                I always get the value error, so I need to spend extra code to handle the issue
                setting ‘Unnamed: 0’ as the index if it contains unique values resetting the index
                otherwise, dropping columns with all NaN values,
                converting the DataFrame to a sparse format,
                and dropping columns that are the same as the first row.
                meanwhile you need to make sure those attributes that you put fit in the function needed

                honestly, it just keeps on trying until you find the accessible method
                making it an ideally dataframe that the algorithm required.

        """
        unique_values = dask_df['Unnamed: 0'].nunique().compute()
        num_rows = dask_df.shape[0].compute()
        if 'Unnamed: 0' in dask_df.columns and unique_values == num_rows:
            dask_df = dask_df.set_index('Unnamed: 0')
        else:
            dask_df = dask_df.reset_index(drop=True)
        pandas_df = dask_df.compute()
        pandas_df = pandas_df.dropna(how='all', axis=1)
        dask_df = dd.from_pandas(pandas_df, npartitions=5)
        dask_df = dask_df.astype(pd.SparseDtype("float", np.nan))
        first_row = dask_df.head(1)
        dask_df = dask_df.loc[:, (dask_df.head(1) != first_row).any()]
        return dask_df


    def load_and_preprocess_data(self, path):
        """
                This function loads a CSV file into a Dask DataFrame,

                Guess what I tried the pandas and the numpy
                their dataframe did not work
                the reason why is that unlike any other filed
                the biology data is always large and those regular dataframe could not handle
                and the dask is build for that
                Sincerely respect the creator of the dask api and the law of biology
                by creating so many beautiful things in our life.

                preprocesses the data using the preprocess_data method,
                and reshapes it if it does not contain a dimension with size 0.
                If the DataFrame has no rows or columns, it skips reshaping.
                Otherwise, it flattens the DataFrame, takes the first 160,000 elements,

                here is one thing, the algorithm will take up a lot of space, RAM,be more specific
                my own computer couldn't afford that,
                and we are the first group student to major the artificial intelligence in the campus,
                Setting the sever, connecting the sever all by myself
                did not have extra energy to consider the more ram that the algorithm required,
                just use a 96G sever to run this algorithm, which is a pity that it could be better.
                More data, better result.
                biology create that prefect feature, but I can't afford that. that's so much pain.
                I found the minion size is around 400,
                so I'm just taking the size of 400*400 since the algorithm require square matrix.

                that just telling that if you have mature sources the output will be better.
                and you need to reset the attributes here.

                reshapes them into a 400x400 array, and converts it back into a Dask DataFrame.

        """
        print(f"Loading and preprocessing data from {path}")
        dask_df = dd.read_csv(path, assume_missing=True, sample=1000000)
        dask_df = self.preprocess_data(dask_df)
        num_rows = len(dask_df)
        num_cols = len(dask_df.columns)
        if num_rows == 0 or num_cols == 0:
            print(f"Skipping reshaping for {path} because it contains a dimension with size 0.")
        else:
            flat_data = dask_df.values.compute().flatten()[:160000]
            flat_data_dask = da.from_array(flat_data)
            reshaped_data = da.reshape(flat_data_dask, (400, 400))
            dask_df = dd.from_dask_array(reshaped_data)

        return dask_df

    def load_and_preprocess_two_datasets2(self, path1, path2, test_size=0.2):
        """
                is made for the causal analysis dataset's split, since you need to merge them
                you need to get rid of one of those index or otherwise that's overwhelmed
                them the rest is the same, just loading and using the previous function then split, it's made for split

        """
        print(f"Loading and preprocessing data from {path1} and {path2}")
        dask_df1 = self.load_and_preprocess_data(path1)
        dask_df2 = self.load_and_preprocess_data(path2)
        print(type(dask_df1))
        print(type(dask_df2))
        print(dask_df1.index.compute())
        print(dask_df2.index.compute())
        pandas_df1 = dask_df1.compute()
        pandas_df1[0] = pandas_df1[0].apply(
            lambda x: int(x.replace('ACH-', '')) if isinstance(x, str) and 'ACH-' in x else x)
        dask_df1 = dd.from_pandas(pandas_df1, npartitions=dask_df1.npartitions)
        dask_df1 = dask_df1.set_index(0)
        merged_df = dask_df1.merge(dask_df2, left_index=True, right_index=True, how='inner')
        train_df, test_df = self.split_data(merged_df, test_size)
        return train_df, test_df

    def load_and_split_data2(self, path, test_size=0.2):
        """
                it just some simple applications of the previous function
        """
        print(f"Loading and preprocessing data from {path}")
        dask_df = self.load_and_preprocess_data(path)
        train_df, test_df = self.split_data(dask_df, test_size)
        return train_df, test_df

    @staticmethod
    def split_data(dask_df, test_size=0.2):
        """
                because the dataset is pretty large, so I am using the hold-out to handle that.
        """
        train_frac = 1 - test_size
        train_df, test_df = dask_df.random_split([train_frac, test_size], random_state=1)
        return train_df, test_df

    def load_and_preprocess_data2(self, path):
        """
                is made for the ultimate dataset
                just some simple cleaning as the brief version of previous functions
        """
        print(f"Loading and preprocessing data from {path}")
        df = pd.read_csv(path)
        df = df.set_index('Unnamed: 0')
        df = df.dropna(axis=1, how='all')
        df = df.loc[:, (df != df.iloc[0]).any()]
        df = df.astype('float')
        return df

    def correlation_analysis(self, data):
        print("Performing correlation analysis")
        data = data.loc[:, data.apply(pd.Series.nunique) != 1]
        data = data.dropna(axis=1, how='all')
        data = data.select_dtypes(include=[np.number])
        print(f"Shape of input data: {data.shape}")
        if data.empty:
            print("No valid data to compute correlation.")
            return []
        else:
            correlations = data.corr().values.flatten().tolist()
            print(f"Length of correlations: {len(correlations)}")
            return correlations

    @staticmethod
    def estimate_causal_effect(args):
        data1, data2, outcome1, outcome2 = args
        X = data1[outcome1]
        y = data2[outcome2]
        y = y.loc[X.index]
        y = y.fillna(0)
        X = X.fillna(X.mean())
        model = SGDRegressor().fit(X.values.reshape(-1, 1), y)
        coef = model.coef_[0]
        return coef


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

        with Pool(5) as pool:
            causal_effects = pool.map(self.estimate_causal_effect, args_generator())
        return causal_effects


if __name__ == "__main__":
    FatherPath = Path(__file__).parents[1]
    CurrentPath = FatherPath / "clear"
    Level1Path = str(CurrentPath / "L1dataset.csv")
    Level2Path = str(CurrentPath / "L2dataset.csv")
    Level3Path = str(CurrentPath / "L3dataset.csv")
    Level4Path = str(CurrentPath / "Drug_sensitivity_AUC_(Sanger_GDSC2)_upgrade.csv")
    my_model = MyModel([Level1Path, Level2Path, Level3Path], [])
    warnings.filterwarnings("ignore", category=FutureWarning)

    correlations = []
    causal_effects = []


    for path in my_model.data_paths:
        dataset = my_model.load_and_preprocess_data(path).compute()
        correlation = my_model.correlation_analysis(dataset)
        correlations.append(correlation)
        print(f"Length of correlations after processing {path}: {len(correlation)}")
        gc.collect()

    causal_effect = my_model.estimate_causal_effects(Level2Path, Level3Path)
    causal_effects.append(causal_effect)
    print(f"Length of causal_effects after processing {Level2Path} and {Level3Path}: {len(causal_effect)}")

    correlations = [torch.tensor(c) for c in correlations if c is not None and len(c) > 0]
    if correlations:
        max_len = max([len(c) for c in correlations])
        correlations = [torch.cat([c, torch.zeros(max_len - len(c))]) for c in correlations]
        correlations = torch.stack(correlations)
    else:
        correlations = torch.tensor([])

    causal_effects = [torch.tensor(c) for c in causal_effects if c is not None and len(c) > 0]
    if causal_effects:
        max_len = max([len(c) for c in causal_effects])
        causal_effects = [torch.cat([c, torch.zeros(max_len - len(c))]) for c in causal_effects]
        causal_effects = torch.stack(causal_effects)
    else:
        causal_effects = torch.tensor([])

    x1_train, x1_test = my_model.load_and_split_data2(Level1Path, test_size=0.2)
    x3_train, x3_test = my_model.load_and_split_data2(Level2Path, test_size=0.2)
    x4_train, x4_test = my_model.load_and_preprocess_two_datasets2(Level2Path, Level3Path, test_size=0.2)
    x5_train, x5_test = my_model.load_and_split_data2(Level3Path, test_size=0.2)


    n_features1 = x1_train.shape[1]
    n_features3 = x3_train.shape[1]
    n_features4 = x4_train.shape[1]
    n_features5 = x5_train.shape[1]

    df = my_model.load_and_preprocess_data2(Level4Path)
    df = df.fillna(df.mean())
    cell_line_mapping = {cell_line: i for i, cell_line in enumerate(df.index.unique())}
    df.index = df.index.map(cell_line_mapping)

    cell_lines = torch.tensor(df.index.values).unsqueeze(1)
    drugs = torch.tensor(df.values)

    n_users = len(df.index.unique())
    n_items = len(df.columns)

    n_corr_features = len(correlations)
    n_causal_features = len(causal_effects)

    model = WideAndDeepModel(n_users, n_items, n_corr_features, n_causal_features, n_features1,n_features3, n_features4,n_features5,
                             correlation=correlations,
                             causal_effect=causal_effects)

    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=0.01)

    cell_lines_train, cell_lines_test = train_test_split(cell_lines, test_size=0.2)
    cell_lines_train, cell_lines_test = map(lambda x: x.clone().detach(), (cell_lines_train, cell_lines_test))
    cell_lines_train2, cell_lines_test2 = cell_lines_train, cell_lines_test
    drugs_train, drugs_test = train_test_split(drugs, test_size=0.2)

    min_length = min(cell_lines_train.shape[0], x1_train.npartitions,  x3_train.npartitions,x4_train.npartitions,x5_train.npartitions)
    cell_lines_train = cell_lines_train[:min_length]

    x1_train = x1_train.head(min_length, compute=True)
    x3_train = x3_train.head(min_length)
    x4_train = x4_train.head(min_length)
    x5_train = x5_train.head(min_length)

    train_data = list(
        zip(cell_lines_train, drugs_train, x1_train, x3_train, x4_train, x5_train, cell_lines_train2))
    train_dl = DataLoader(train_data, batch_size=32)

    test_data = list(zip(cell_lines_test, drugs_test, x1_test, x3_test, x4_test, x5_test, cell_lines_test2))
    test_dl = DataLoader(test_data, batch_size=32)


    for epoch in range(10):
        for (cell_lines, drugs, x1, x3, x4, x5, labels) in train_dl:
            print("Type of 'celllines':", type(cell_lines))
            print("Value of 'celllines':", cell_lines)
            print("Type of 'drug':", type(drugs))
            print("Value of 'drug':", drugs)
            print("Type of 'x1':", type(x1))
            print("Value of 'x1':", x1)
            print("Type of 'x3':", type(x3))
            print("Value of 'x3':", x3)
            print("Type of 'x4':", type(x4))
            print("Value of 'x4':", x4)
            print("Type of 'x5':", type(x5))
            print("Value of 'x5':", x5)
            #!
            outputs = model(cell_lines.long(), drugs.long(), float(x1[0]), float(x3[0]), float(x4[0]),
                            float(x5[0]))
            labels = labels.float()
            if outputs.shape != labels.shape:
                outputs = outputs.view(labels.shape)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    x1_test_tensor = torch.tensor(x1_test.compute().values).float()

    x3_test_tensor = torch.tensor(x3_test.compute().values).float()
    x4_test_tensor = torch.tensor(x4_test.compute().values).float()
    x5_test_tensor = torch.tensor(x5_test.compute().values).float()
    cell_lines_test = cell_lines_test.unsqueeze(0) if cell_lines_test.dim() == 1 else cell_lines_test


    with torch.no_grad():
        loss = None
        for (cell_lines, drugs, x1, x3, x4, x5, labels) in test_dl:
            outputs = model(cell_lines.long(), drugs.long(), float(x1[0]), float(x3[0]), float(x4[0]),
                            float(x5[0]))
            labels = labels.float()
            if outputs.shape != labels.shape:
                outputs = outputs.view(labels.shape)
            loss = criterion(outputs, labels)
        if loss is None:
            print('Test Loss: No data in test_dl')
        elif torch.is_tensor(loss):
            print(f'Test Loss: {loss.item()}')


    with torch.no_grad():
        for (cell_lines, drugs, x1, x3, x4, x5, labels) in test_dl:
            outputs = model(cell_lines.long(), drugs.long(), float(x1[0]), float(x3[0]), float(x4[0]),
                            float(x5[0]))
            labels = labels.float()
            if outputs.shape != labels.shape:
                outputs = outputs.view(labels.shape)
            loss = criterion(outputs, labels)
        if torch.is_tensor(loss):
            print(f'Test Loss: {loss.item()}')
        else:
            print('Test Loss: No data in test_dl')
