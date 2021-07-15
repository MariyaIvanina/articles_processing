import pandas as pd
import os
from time import time

class ExcelReader:

    def __init__(self):
        pass

    def read_file(self, filename):
        df = pd.DataFrame()
        if filename.split(".")[-1] == "csv":
            try:
                df = pd.read_csv(filename).fillna("")
            except:
                df = pd.read_csv(filename, encoding='ISO-8859-1').fillna("")
        if filename.split(".")[-1] == "xlsx":
            df = pd.read_excel(filename).fillna("")
        if len(df) == 0:
            print("filename %s has incorrect extension"%filename)
        assert len(df) != 0
        return df

    def read_df_from_excel(self, filename):
        start_time = time()
        articles_df_saved = self.read_file(filename)
        print("Read file %s: %.2fs"%(filename, time() - start_time))
        start_time = time()
        for i in range(len(articles_df_saved)):
            for column in articles_df_saved.columns:
                try:
                    if type(articles_df_saved[column].values[i]) != str:
                        continue
                    val = articles_df_saved[column].values[i].strip()
                    if val[0] == "[" and val[-1] == "]":
                        result = eval(articles_df_saved[column].values[i])
                        if type(result) == list:
                            articles_df_saved[column].values[i] = result
                except:
                    pass
        print("Processed file %s: %.2fs"%(filename, time() - start_time))
        return articles_df_saved

    def read_distributed_df_from_excel(self, folder):
        files_count = len(os.listdir(folder))
        full_df = pd.DataFrame()
        for filename_id in range(files_count):
            temp_df = self.read_df_from_excel(os.path.join(folder, "%d.xlsx"%filename_id))
            full_df = pd.concat([full_df, temp_df], sort=False)
        return full_df

    def read_df(self, file_folder):
        if os.path.isdir(file_folder):
            try:
                return self.read_distributed_df_from_excel(file_folder)
            except:
                return self.read_folder(file_folder)
        return self.read_df_from_excel(file_folder)

    def read_distributed_df_sequantially(self, folder):
        files_count = len(os.listdir(folder))
        for filename_id in range(files_count):
            yield self.read_df_from_excel(os.path.join(folder, "%d.xlsx"%filename_id))

    def read_folder(self, folder):
        full_df = pd.DataFrame()
        for filename in os.listdir(folder):
            temp_df = self.read_df_from_excel(os.path.join(folder, filename))
            full_df = pd.concat([full_df, temp_df], sort=False)
        return full_df
