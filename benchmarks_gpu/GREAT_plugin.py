from synthcity.plugins.core.plugin import Plugin
from synthcity.plugins.core.distribution import Distribution
from synthcity.plugins.core.schema import Schema
from synthcity.plugins.core.dataloader import DataLoader
from synthcity.plugins.generic.plugin_great import GReaTPlugin

import pandas as pd
import os
from typing import Any, List

import sys 
sys.path.insert(0, './baselines/be_great_pafted/be_great/')
from be_great import GReaT


class GREAT_plugin(Plugin):
    def __init__(
        self,
        exp_path,
        train=True,
        batch_size=32,
        epochs=50,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        # self.sampling_patience = 1
        # self.strict = False
        # self.device = 'cuda'
        self.exp_path = exp_path
        self.train = train
        self.batch_size = batch_size
        self.epochs = epochs

        if self.train:
            self.model = GReaT(llm='distilgpt2', experiment_dir=f"{self.exp_path}/trainer_great",
                               batch_size=self.batch_size, epochs=self.epochs)
        else:
            self.model = GReaT.load_from_dir(f"{self.exp_path}/models/")
                
    @staticmethod
    def name() -> str:
        return "great"

    @staticmethod
    def type() -> str:
        return "debug"

    @staticmethod
    def hyperparameter_space(**kwargs: Any) -> List[Distribution]:
        """
        We can customize the hyperparameter space, and use it in AutoML benchmarks.
        """
        return []

    def _fit(self, X: DataLoader, *args: Any, **kwargs: Any):
        # self._data_encoders = None
        
        if self.train:
            train_data = X.dataframe()
            self.model.fit(train_data, having_order=True)
            
            model_path = f"{args.exp_path}/models/"
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            
            self.model.save(model_path)

        return self #already trained

    def _generate(self, count: int, syn_schema: Schema, **kwargs: Any) -> pd.DataFrame:
        return self._safe_generate(self.model.sample, count, syn_schema, device=self.device)
    
class PAFT_plugin(Plugin):
    def __init__(
        self,
        order_dict_path,
        exp_path,
        train=True,
        batch_size=32,
        epochs=50,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        # self.sampling_patience = 1
        # self.strict = False
        # self.device = 'cuda'
        self.exp_path = exp_path
        self.train = train
        self.batch_size = batch_size
        self.epochs = epochs

        if self.train:
            self.model = GReaT(llm='distilgpt2', experiment_dir=f"{self.exp_path}/trainer_great",
                               batch_size=self.batch_size, epochs=self.epochs)
        else:
            self.model = GReaT.load_from_dir(f"{self.exp_path}/models/")
                
    @staticmethod
    def name() -> str:
        return "paft"

    @staticmethod
    def type() -> str:
        return "debug"

    @staticmethod
    def hyperparameter_space(**kwargs: Any) -> List[Distribution]:
        """
        We can customize the hyperparameter space, and use it in AutoML benchmarks.
        """
        return []

    def _fit(self, X: DataLoader, *args: Any, **kwargs: Any):
        # self._data_encoders = None
        
        if self.train:
            train_data = X.dataframe()
            self.model.fit(train_data, having_order=True)
            
            model_path = f"{args.exp_path}/models/"
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            
            self.model.save(model_path)

        return self #already trained

    def _generate(self, count: int, syn_schema: Schema, **kwargs: Any) -> pd.DataFrame:
        return self._safe_generate(self.model.sample, count, syn_schema, device=self.device)

class PaftPlugin(GReaTPlugin):
    def __init__(
        self,
        **kwargs
        ) -> None:
        super().__init__(**kwargs)
                
    @staticmethod
    def name() -> str:
        return "paft"