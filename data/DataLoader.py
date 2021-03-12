from abc import ABC, abstractmethod
import os
import pandas as pd

class DataLoader(ABC):
    
    def __init__(self, path):
        self.path = path
    
    @classmethod
    def get_loader_instance(cls, path):
        if path == os.path.join("data", "ph_1628"):
            return DataLoader1628( os.path.join(path,"data_selected") )
        elif path == os.path.join("data", "ph_names"):
            return DataLoaderNames( os.path.join(path,"data_selected") )
    
    def get_num_datasets(self):
        return len(self.get_dataset_list())
    
    def get_num_objectives(self):
        return len(self.get_objective_names())
    
    @abstractmethod
    def get_objectives(self):
        pass
    
    def get_datasets(self):
        files = [f for f in os.listdir(self.path) if f.endswith('.csv')]
        return files
    
    def load_dataset(self, name, obj):
        df =  pd.read_csv(os.path.join(self.path, name))
        # idx = self.get_objectives().index(obj)
        X = df.drop([obj], axis = 1)
        y = df[obj]
        return X, y

class DataLoader1628(DataLoader):
    
    def get_objectives(self):
        return [
            "monthly_commits",
            "monthly_contributors",
            "monthly_stargazer",
            "monthly_open_PRs",
            "monthly_closed_PRs",
            "monthly_open_issues",
            "monthly_closed_issues"
            ]
    
    def load_dataset(self, name, obj):
        X, y = super().load_dataset(name, obj)
        X = X.drop(["dates"], axis = 1)
        return X, y

class DataLoaderNames(DataLoader1628):
    def get_objectives(self):
        return [
            "number_of_commits",
            "number_of_contributors",
            "number_of_new_contributors",
            "number_of_contributor-domains",
            "number_of_new_contributor-domains",
            "number_of_open_PRs",
            "number_of_closed_PRs",
            "number_of_open_issues",
            "number_of_closed_issues",
            "number_of_stargazers"
            ]

