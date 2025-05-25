
import itertools
import time
import pandas as pd
import numpy as np

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, MinMaxScaler
import torch 
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



# TODO determine when this gets run
SEED = 123
torch.manual_seed(SEED)

DEFAULT_DTYPE=torch.float32
# DEFAULT_DTYPE=float

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


class NormalizedClassifierDatasetMetadata:

    ## TODO have metadata for the embedding dims

    def __init__(self, label_column):
        self.label_column = label_column
        self.categorical_map = {}
        self.ordinal_map = {}
        self.ordinal_numeric_cols = []
        self.embedding_cols = []

    def set_categorical_map(self, categorical_map):
        self.categorical_map = categorical_map

    def set_ordinal_map(self, ordinal_map):
        self.ordinal_map = ordinal_map

    ## todo provide a helper that gets the numerics from a dataframe
    def set_ordinal_numeric_cols(self, ordinal_numeric_cols):
        self.ordinal_numeric_cols = ordinal_numeric_cols

    def set_embedding_cols(self, embedding_cols):
        self.embedding_cols = embedding_cols

    def get_columns(self):
        return (
            [self.label_column]
            + list(self.categorical_map.keys())
            + list(self.ordinal_map.keys()) 
            + self.ordinal_numeric_cols 
            + self.embedding_cols            
        )

class NormalizedClassifierDataset (torch.utils.data.Dataset):

    def __init__(self, orig_df, ds_meta):      
        df_copy = orig_df.copy()

        # first step, only select the columns defined in ds_meta

        df_copy = df_copy[ ds_meta.get_columns() ]

        # first, set aside the labels
        self.labels_ndarray = torch.tensor(df_copy.pop(ds_meta.label_column).values, dtype=DEFAULT_DTYPE)

        # prepare a OneHotEncoder for values in the categorical_map
        mapped_values_array = list(ds_meta.categorical_map.values())
        mapped_cols = ds_meta.categorical_map.keys()
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='error', categories=mapped_values_array)
        ohe = ohe.fit( df_copy[mapped_cols] )
        # apply it to the df using the new names from the encoder
        newFeatureColsForEncoding = ohe.get_feature_names_out()
        df_copy[ newFeatureColsForEncoding ] = ohe.transform( df_copy[mapped_cols] )
        # # clean up orig we've encde
        df_copy.drop( mapped_cols, axis=1, inplace=True)

        # now deal with the ordinals. use the rankings from the map to apply an 
        # OrdinalEncoder in place one column at a time
        for col, ordered_categories in ds_meta.ordinal_map.items():
            col_ordinal_encoder = OrdinalEncoder(categories=[ordered_categories])
            df_copy[col] = col_ordinal_encoder.fit_transform( df_copy[ [col] ] )

        # next scale the numeric ordinal columns
        if len(ds_meta.ordinal_numeric_cols) > 0:
            scaler = MinMaxScaler()
            df_copy[ds_meta.ordinal_numeric_cols] = scaler.fit_transform(df_copy[ds_meta.ordinal_numeric_cols])

        # here we deal with embeddings
        embedding_values = []
        if len(ds_meta.embedding_cols) > 0:  
            for col in ds_meta.embedding_cols:
                # first, let's create an index for each col value
                col_value_index = {val.item(): idx for idx, val in enumerate( df_copy[col].fillna(0).unique() )}
                indexesForColumn = df_copy[col].map(col_value_index)
                num_embeddings = len(col_value_index) # have embeddings for each key
                embedding_dim = 2 # int(max(np.ceil(num_embeddings ** 0.25), 4)) # use "fourth root" rule of thumb 
                col_embedding_layer = nn.Embedding(num_embeddings, embedding_dim, dtype=DEFAULT_DTYPE)

                colEmbeddings = col_embedding_layer( torch.tensor(indexesForColumn.to_numpy())  )
                embedding_values.append(colEmbeddings)

        # let's finalize the contents of the dataframe and get it into tensor
        features_tensor = torch.tensor(df_copy.to_numpy(), dtype=DEFAULT_DTYPE)
        
        # and cat the results of embeddings onto it
        combined_tensor = torch.cat( (features_tensor, *embedding_values), dim=1)
        self.features_ndarray = combined_tensor
        
    def __len__(self):
        return self.features_ndarray.shape[0]
            
    def __getitem__(self, idx):

        features_tensor = self.features_ndarray[idx]
        label_tensor = self.labels_ndarray[idx]

        return features_tensor, label_tensor 
    
    def get_feature_count(self):
        ## need to clone/detach/do ritual sacrifice so it doesnt warn about needing grad externally
        return len(self.features_ndarray[0].clone().detach().requires_grad_(True))
    

class GeneralNN(nn.Module):
    def __init__(self, inputFeatures, layerInputs, dropOutRate, outputLayer=nn.Sigmoid()):
        super().__init__()
        
        stack = []
        for fromLayer, toLayer in itertools.pairwise([inputFeatures] + layerInputs):
            stack += [
                nn.Linear(fromLayer, toLayer),
                nn.GELU(),
                nn.Dropout(p=dropOutRate), 
            ]

        # TODO consider a more pythonic way to clean up the output layers
        # pop the last activation and nn.Dropout() going into the output layer
        stack.pop()
        stack.pop()

        stack.append(outputLayer)
        self.linear_relu_stack = nn.Sequential(*stack).to(dtype=DEFAULT_DTYPE)

    def forward(self, x):
        return self.linear_relu_stack(x)


class TrainingManager:    
    ## TODO override tostring to print metrics

    def __init__(self, model):
        self.model = model

    def train(self, dataloader, num_epochs):

        loss_fn   = nn.BCELoss()  # binary cross entropy
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        for epoch in range(num_epochs):

            epochStart = time.perf_counter()

            self.model.train()
            epoch_running_loss, epoch_running_correct, epoch_running_guesses = 0.0, 0, 0

            for X, y in dataloader:

                # X, y = X.to(device), y.to(device)
                optimizer.zero_grad()

                y_pred = self.model(X)
                y_pred = y_pred.reshape(y.shape)

                y_pred_guess = torch.round(y_pred)
                batch_num_correct = (y == y_pred_guess).sum()
                batch_guesses = len(y_pred_guess)

                loss = loss_fn(y_pred, y)
                loss.backward()
                optimizer.step()
                
                epoch_running_correct += batch_num_correct
                epoch_running_guesses += batch_guesses
                epoch_running_loss += loss.item() * batch_guesses

            epoch_avg_loss = epoch_running_loss / len(dataloader.dataset)
            print( (
                f'Epoch [{epoch+1}/{num_epochs}], '
                f'Avg training Loss: {epoch_avg_loss:.4f}, '
                f'Accuracy: {epoch_running_correct}/{epoch_running_guesses} ({epoch_running_correct/epoch_running_guesses:.4f}), '
                f'in {time.perf_counter() - epochStart:.2f} seconds.'
            ))
    
    # takes an array of dataframes and an encoder
    def eval(self, test_dataloader):
        self.model.eval()  # Set the model to evaluation mode

        # Define your loss function (e.g., Binary Cross-Entropy)
        criterion = nn.BCEWithLogitsLoss()  # Commonly used for binary classification

        # Lists to store results
        test_losses = []
        all_preds = []
        all_labels = []

        # Iterate over the test batches
        with torch.no_grad():  # Disable gradient calculations
            for inputs, labels in test_dataloader:

                # just make this an array of the labels (instead array of arrays with one element)
                labels = labels.reshape(-1)

                y_pred = self.model(inputs)
                y_pred = y_pred.reshape(labels.shape)  # make sure it matches

                loss = criterion(y_pred, labels)
                test_losses.append(loss.item())

                y_pred_guess = torch.round(y_pred)
                
                all_preds.extend(y_pred_guess.numpy())
                all_labels.extend(labels.numpy())

        # Calculate overall metrics
        avg_test_loss = np.mean(test_losses)
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, zero_division=np.nan)

        print(f"Average Test Loss: {avg_test_loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test Precision: {precision:.4f}")
        print(f"Test Recall: {recall:.4f}")
        print(f"Test F1-Score: {f1:.4f}")

