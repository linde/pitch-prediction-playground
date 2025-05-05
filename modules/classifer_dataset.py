


import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
import torch 


## TODO make this work as vargs
class ClassifierDataset (torch.utils.data.Dataset):    
    def __init__(self, raw_df, label_column):
        df_copy = raw_df.copy()

        # first, set aside the labels
        self.labels_ndarray = df_copy.pop(label_column).values

        # TODO consider asserting about dtypes (ie only numerics at this point)

        # process columns to normalize values 
        scaler = preprocessing.MinMaxScaler()
        self.features_ndarray = scaler.fit_transform(df_copy)
        
    def __len__(self):
        return self.features_ndarray.shape[0]
            
    def __getitem__(self, idx):
        
        features_tensor = torch.tensor(self.features_ndarray[idx], dtype=torch.float32)
        label_tensor = torch.tensor(self.labels_ndarray[idx], dtype=torch.float32)

        return features_tensor, label_tensor 
    
    def get_feature_count(self):
        return len(self.features_ndarray[0])


    # takes an array of dataframes and an encoder
    @staticmethod
    def onehot_encode_datafames(df_array):
        unioned_df = pd.concat(df_array)
        union_categorical_cols = unioned_df.select_dtypes(exclude=['number']).columns
        
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore', )
        ohe = ohe.fit( unioned_df[union_categorical_cols] )

        processed_df_array = []
        for df in df_array:
            # first transform using the encoder that was fit on the unioned data
            encoded_values = ohe.transform( df[union_categorical_cols] )
            # and make a dataframe from that
            encoded_value_features = ohe.get_feature_names_out()
            encoded_df = pd.DataFrame(encoded_values, columns=encoded_value_features)
            # drop the encoded features from the original df and concat the encodings
            df_processed = pd.concat([df.drop(columns=union_categorical_cols), encoded_df], axis=1)
            # and finally, set it aside
            processed_df_array.append(df_processed)

        return processed_df_array
