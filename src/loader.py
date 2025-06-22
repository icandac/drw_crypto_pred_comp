import numpy as np
import dask.dataframe as dd

class DataLoader:
    @staticmethod
    def downcast_numeric_dask(df):
        float_cols = df.select_dtypes(include='float64').columns
        for col in float_cols:
            df[col] = df[col].astype('float32')
        int_cols = df.select_dtypes(include='int64').columns
        for col in int_cols:
            df[col] = df[col].astype('int32')
        return df

    def load_and_preprocess(self, path, start_date="2023-12-01", columns=None):
        df = (
            dd.read_parquet(path, columns=columns)
            .loc[start_date:]
            .replace(-np.inf, np.nan)
            .map_partitions(self.downcast_numeric_dask)
            .compute()
        )
        return df
