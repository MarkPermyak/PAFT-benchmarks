import os
import sys 
from be_great_pafted import GReaT as GReaT_pafted
from be_great.great import GReaT as GReaT_og

import argparse

# third party
import pandas as pd
import random


def data_exist(train_folder, test_folder, df_name) -> bool:
    train_exist = os.path.exists(f'{train_folder}/{df_name}.csv')
    test_exist  = os.path.exists(f'{test_folder}/{df_name}.csv')

    if not train_exist:
        print(f'No train data for dataset "{df_name}"')
    
    if not test_exist:
        print(f'No test data for dataset "{df_name}"')

    return train_exist and test_exist

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Baseline Runner")
    parser.add_argument("--data_folder", default=f"./data", help="Data Folder")
    parser.add_argument("--output_folder", default=f"./results", help="Data Folder")
    parser.add_argument("--repeats", default=5, type=int, help="Number of experiments")
    parser.add_argument("--no_rows_filter", action='store_true', help="Include dfs with rows >= 50k")
    
    args = parser.parse_args()

    data_folder = args.data_folder
    output_folder = args.output_folder

    train_folder = f'{data_folder}/train'
    test_folder = f'{data_folder}/test'

    if not os.path.exists(f'{data_folder}/data_info.csv'):
        raise Exception(f'Missing Targets df in folder "{data_folder}"')

    if not os.path.exists('feature_order_permutation.txt'):
        raise Exception('Need to calculate feature order for PAFT benchmarks')
    
    with open('./feature_order_permutation.txt', 'r') as f:
        sorted_order_dict = eval(f.read())
        

    plugins_list = ['great', 'paft']

    # Sort dfs according to row number
    data_info = pd.read_csv(f'{data_folder}/data_info.csv', index_col='df_name').sort_values('row_number')

    if not args.no_rows_filter:
        data_info = data_info[data_info['row_number'] <= 50_000]

    # df_names = data_info.index

    # problem_dfs = [
    #     'wine', 
    #     '560_bodyfat'
    # ]

    df_names = [
        'diabetes',
        'breast_cancer',
        'credit_g',
        'phoneme',
        'page_blocks',
        '562_cpu_small',
        'heloc_dataset_v1',
        'pendigits',
        'online_shoppers_intention',
        'nursery',
        'magic04',
        # 'default_of_credit_card_clients',
        # 'OnlineNewsPopularity',
        ]
    # df_names = [a for a in df_names if 'wine' not in a]

    for df_name in df_names:
        if df_name == 'OnlineNewsPopularity':
            continue
            
        print(f'Start "{df_name}"')

        if not data_exist(train_folder, test_folder, df_name):
            print(f'"{df_name}": No train or test, skipped')
            continue
        
        df_train = pd.read_csv(f'{train_folder}/{df_name}.csv')
        df_test = pd.read_csv(f'{test_folder}/{df_name}.csv')

        target_name = data_info.loc[df_name, 'target_name']
        task_type   = data_info.loc[df_name, 'task_type']
        df_test_len = data_info.loc[df_name, 'df_test_len']

        if len(df_train) <= 10_000:
            epochs = 100
        else:
            epochs = 50

        for plugin_name in plugins_list:
            # reorder for paft plugin
            if plugin_name == 'paft':
                order = sorted_order_dict[df_name]

                if order == '':
                    # No df in calculated func dependencies
                    order = df_train.columns.tolist() # a fixed random order for all samples
                    # random order
                    random.shuffle(order)
                    order = ','.join(order)
                print(df_name, ":", order)
                df_train = df_train[order.split(',')]
            
            print(f'Start training plugin "{plugin_name}"')
            
            if plugin_name == 'great':
                model = GReaT_og(
                    llm='distilgpt2',
                    epochs=epochs, 
                    efficient_finetuning='lora', 
                    **{'report_to' : 'none', 'save_strategy' : 'no'})
            
            if plugin_name == 'paft':
                model = GReaT_pafted(
                    llm='distilgpt2', 
                    epochs=epochs, 
                    efficient_finetuning='lora', 
                    **{'report_to' : 'none', 'save_strategy' : 'no'})
                
            
            model.fit(df_train)
            print(plugin_name, 'fitted OK')
            

            model_path = f'./great_models/{df_name}/{plugin_name}/'
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            model.save(model_path)
            print(plugin_name, 'model saved OK')

            # Start of experiments
            generated_data_folder = args.output_folder + f'/generated_data/{df_name}/{plugin_name}/'
            os.makedirs(os.path.dirname(generated_data_folder), exist_ok=True)

            for repeat in range(args.repeats):
                X_syn = model.sample(df_test_len)

                repeat_save_path = generated_data_folder + f'/X_syn_{repeat}.csv'
                X_syn.to_csv(repeat_save_path, index=False)

                print(plugin_name, repeat, 'generated and saved OK')
                
        print(df_name, ', Done\n')