import argparse
import os
import yaml
from common.utils import load_all_data_encoder
from sklearn.model_selection import KFold
from common.utils import scorePredict
from model.model import Model
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import json
from sklearn.utils.class_weight import compute_class_weight



def main(parser):
    args = parser.parse_args()
    model_dir = args.model_dir
    label_to_exclude = args.label_to_exclude
    is_sweeping = args.is_sweeping
    is_cross_validation = args.is_cross_validation
    best_result_config = None
    config_path = args.config_path
    model_arg = args.model_arg

    with open(os.getcwd() + config_path) as f:
        config = json.load(f)
    df_model_args = pd.read_json(os.getcwd() + model_arg)
    model_args = df_model_args.to_dict(orient='records')[0]
    model_name = config["model_name"]
    if model_dir != "":
        model_name = os.getcwd() + model_dir

    df_test, df_test_values = load_all_data_encoder(config["test_file"], label_to_exclude, config["label"],
                                            config["filter_label"], config["filter_label_value"])

    labels = list(df_test['labels'].unique())
    labels_list = labels
    if config["is_training"]:
        df_train, df_train_values = load_all_data_encoder(config["train_file"], label_to_exclude, config["label"],
                                                  config["filter_label"], config["filter_label_value"])
        labels = list(df_train['labels'].unique())
        ############### Calculate weights using sklearn ##################
        if "weight" in df_model_args:
            # weights = compute_class_weight(np.unique(df_train['labels'].values), list(df_train['labels'].values))
            weights = compute_class_weight(class_weight="balanced", classes=np.unique(df_train['labels'].values),
                                           y=df_train['labels'].values)
            model_args["weight"] = list(weights)

        labels_list = labels
        wandb_config = {}
        if is_cross_validation:
            model = Model(config["model_type"], model_name, config["use_cuda"], len(labels), config["wandb_project"],
                          wandb_config,
                          is_sweeping, config["is_evaluate"], best_result_config, config["is_training"], value_head=0,
                          output_dir=os.getcwd() + '/models/' + config["output_dir"], model_args=model_args)
            n = 5
            kf = KFold(n_splits=n, random_state=3, shuffle=True)
            results = []
            for train_index, val_index in kf.split(df_train):
                train_df = df_train.iloc[train_index]
                val_df = df_train.iloc[val_index]
                acc = model.train_model(train_df, eval_df=val_df)
                results.append(acc)
            print("results", results)
            print(f"Mean-Precision: {sum(results) / len(results)}")
        else:
            df_eval = None
            stance_model = Model(config["model_type"], model_name, config["use_cuda"], len(labels),
                                 config["wandb_project"], wandb_config,
                                 is_sweeping, config["is_evaluate"], best_result_config, config["is_training"],
                                 value_head=0,
                                 output_dir=os.getcwd() + '/models/' + config["output_dir"], model_args=model_args)
            stance_model.fit(df_train, df_eval=df_eval)

    else:
        stance_model = Model(config["model_type"], model_name, config["use_cuda"], labels_len=3)

    y_pred, model_outputs_test = stance_model.predict_task(df_test)
    y_pred = np.argmax(model_outputs_test, axis=1)

    df_pred = pd.DataFrame(y_pred, columns=['labels'])
    labels_test = pd.Series(df_test['labels']).to_numpy()
    labels = list(df_test['labels'].unique())
    labels.sort()
    result, f1 = scorePredict(labels_test, df_pred.values, labels)
    print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--config_path",
                        default="/data/VIL_1/vil.json",
                        type=str,
                        help="File path to configuration parameters.")

    parser.add_argument("--model_arg",
                        default="/data/VIL_1/vil_model.json",
                        type=str,
                        help="File path to model configuration parameters.")

    parser.add_argument("--model_dir",
                        default="",
                        type=str,
                        help="This parameter is the relative dir of model for predict.")

    parser.add_argument('--label_to_exclude',
                        default=[],
                        nargs='+',
                        help="This parameter should be used if you want to execute experiments with fewer classes.")

    parser.add_argument("--is_sweeping",
                        default=False,
                        action='store_true',
                        help="This parameter should be True if you use sweep search.")

    parser.add_argument("--is_cross_validation",
                        default=False,
                        action='store_true',
                        help="This parameter should be True if you want to make a cross-validation.")

    main(parser)
