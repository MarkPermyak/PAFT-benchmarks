import argparse
import os 
from ast import literal_eval

# stdlib
import sys
import random
import warnings
warnings.filterwarnings("ignore")

# synthcity absolute
import synthcity.logger as log
from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader

# stdlib
from typing import Any, List

# third party
import numpy as np
import pandas as pd

# synthcity absolute
from synthcity.plugins.core.dataloader import GenericDataLoader


# synthcity absolute
from synthcity.plugins import Plugins
from GREAT_plugin import GREAT_plugin, PAFT_plugin, PaftPlugin
from GANBLR_plugin import GANBLR_plugin, GANBLRPP_plugin

# synthcity absolute
from synthcity.metrics import Metrics
from synthcity.metrics.scores import ScoreEvaluator

import pickle

def data_exist(train_folder, test_folder) -> bool:
    train_exist = os.path.exists(f'{train_folder}/{df_name}.csv')
    test_exist  = os.path.exists(f'{test_folder}/{df_name}.csv')

    if not train_exist:
        print(f'No train data for dataset "{df_name}"')
    
    if not test_exist:
        print(f'No test data for dataset "{df_name}"')

    return train_exist and test_exist

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Baseline Runner")
    # parser.add_argument("--model", default=f"paft", help="model name")
    # parser.add_argument("--train", action="store_true", help="Enable train mode")
    # parser.add_argument("--order", default=None, type=str, help="Wanted order")
    # parser.add_argument("--epochs", default=50, type=int, help="Number of epochs")
    parser.add_argument("--repeats", default=5, type=int, help="Number of experiments")
    parser.add_argument("--data_folder", default=f"./data", help="Data Folder")
    
    args = parser.parse_args()

    data_folder = args.data_folder
    train_folder = f'{data_folder}/train'
    test_folder = f'{data_folder}/test'

    if not os.path.exists(f'{data_folder}/data_info.csv'):
        raise Exception(f'Missing Targets df in folder "{data_folder}"')


    # df_names_with_csv = os.listdir(train_folder)
    # for file_name in df_names_with_csv:
    #     if not file_name.endswith('.csv'):
    #         raise Exception('Only files *.csv must be in train folder')
    

    init_kwargs = {
        'great' : {'n_iter' : 10, 'train_kwargs' : {'efficient_finetuning' : 'lora', 'save_strategy' : 'no'}},
        'paft' : {'n_iter' : 5},
        'ctgan' : {'n_iter' : 10},
        'tvae' : {'n_iter' : 10},
        'rtvae' : {'n_iter' : 10},
        'ddpm' : {'n_iter' : 10},
        'ganblr' : {},
        'ganblr++' : {'random_state' : 239, 
                    #   'numerical_columns' : [...] Different for every dataset
                      },
        'pategan' : {'n_iter' : 10},
        'adsgan' : {'n_iter' : 10},
        'bayesian_network' : {'struct_learning_n_iter' : 10}
    }

    fit_kwargs = {
        'great' : {},
        'paft' : {},
        'ctgan' : {},
        'tvae' : {},
        'rtvae' : {},
        'ddpm' : {},
        'ganblr' : {'epochs' : 10, 'verbose' : None},
        'ganblr++' : {'epochs' : 10, 'verbose' : None},
        'pategan' : {},
        'adsgan' : {},
        'bayesian_network' : {'struct_learning_n_iter' : 10}
    }

    metrics =  {
            'sanity': ['data_mismatch', 'common_rows_proportion', 'nearest_syn_neighbor_distance', 'close_values_probability', 'distant_values_probability'],
            'stats': ['jensenshannon_dist', 'chi_squared_test', 'feature_corr', 'inv_kl_divergence', 'ks_test', 'max_mean_discrepancy', 'wasserstein_dist', 'prdc', 'alpha_precision', 'survival_km_distance'],
            'performance': ['linear_model', 'mlp', 'xgb', 'feat_rank_distance'],
            'detection': ['detection_xgb', 'detection_mlp', 'detection_gmm', 'detection_linear'],
            'privacy': ['delta-presence', 'k-anonymization', 'k-map', 'distinct l-diversity', 'identifiability_score']
    }

    plugins_list = [
                # 'great',
                # 'paft',
                'ctgan', 'tvae', 'rtvae', 'ddpm', 'ganblr', 'ganblr++', 'pategan', 'adsgan', 'bayesian_network']
    
    generators = Plugins()

    if 'paft' in plugins_list:
        if not os.path.exists('feature_order_permutation.txt'):
            raise Exception('Need to calculate feature order for PAFT benchmarks')
        
        with open('./feature_order_permutation.txt', 'r') as f:
            sorted_order_dict = eval(f.read())

        generators.add('paft', PaftPlugin)
        

    if 'ganblr' in plugins_list:
        generators.add('ganblr', GANBLR_plugin)
    
    if 'ganblr++' in plugins_list:
        generators.add('ganblr++', GANBLRPP_plugin)


    print('Plugins Added')

    # Sort dfs according to row number
    data_info = pd.read_csv(f'{data_folder}/data_info.csv', index_col='df_name').sort_values('row_number')
    df_names = data_info.index

    df_names = ['iris', 'kekes', '560_bodyfat']
    plugins_list = ['ctgan', 'tvae']

    for df_name in df_names:
        print('Start', df_name)

        if not data_exist(train_folder, test_folder):
            print(f'{df_name}, Benchmarks skipped')
            continue
        
        df_train = pd.read_csv(f'{train_folder}/{df_name}.csv')
        df_test = pd.read_csv(f'{test_folder}/{df_name}.csv')

        target_name = data_info.loc[df_name, 'target_name']
        task_type   = data_info.loc[df_name, 'task_type']

        if 'ganblr++' in plugins_list:
            init_kwargs['ganblr++']['numerical_columns'] = literal_eval(data_info.loc[df_name, 'numeric_cols_indxs'])
        
        benchmarks_results = {}

        for plugin_name in plugins_list:
            if plugin_name == 'paft':
                order = sorted_order_dict[df_name]

                if order == '':
                    # No df in calculated func dependencies
                    order = df_train.columns.tolist() # a fixed random order for all samples
                    # random order
                    random.shuffle(order)
                    order = ','.join(order)

                train_loader = GenericDataLoader(df_train[order.split(',')], target_column=target_name)
                test_loader  = GenericDataLoader(df_test[order.split(',')], target_column=target_name)            
            else:
                train_loader = GenericDataLoader(df_train, target_column=target_name)
                test_loader  = GenericDataLoader(df_test, target_column=target_name)

            print(plugin_name, 'Start')
            
            gen = generators.get(plugin_name, **init_kwargs[plugin_name])
            gen.fit(train_loader, **fit_kwargs[plugin_name])

            print(plugin_name, 'fitted OK')

            scores = ScoreEvaluator()

            for repeat in range(args.repeats):
                X_syn = gen.generate(len(df_test))

                print(plugin_name, repeat, 'generated OK')

                experiment_scores = Metrics.evaluate(
                            X_gt=test_loader,
                            X_syn=X_syn,
                            # X_train=loader,
                            metrics=metrics,
                            task_type=task_type,
                            random_state=239 * repeat,
                            )
        
        
                mean_score = experiment_scores["mean"].to_dict()
                errors = experiment_scores["errors"].to_dict()
                duration = experiment_scores["durations"].to_dict()
                direction = experiment_scores["direction"].to_dict()

                for key in mean_score:
                    scores.add(
                        key,
                        mean_score[key],
                        errors[key],
                        duration[key],
                        direction[key],
                    )

                print(repeat, 'experiment OK')
            
            benchmarks_results[plugin_name] = scores.to_dataframe()



        output_folder = f'./synthcity_res/{df_name}/result_dict.pckl'
        os.makedirs(os.path.dirname(output_folder), exist_ok=True)

        with open(output_folder, 'wb+') as f:
            pickle.dump(benchmarks_results, f)

        print(df_name, ', Benchmarks results saved')