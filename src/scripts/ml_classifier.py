import argparse
import os
from common.utils import load_all_data_encoder
from common.utils import scorePredict
from common.features_extraction import create_tf_idf_model, convert_text_to_features
import json
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB


models = [
          SVC(kernel="linear", C=0.025),
          RandomForestClassifier(random_state=1, max_depth=5),
          LogisticRegression(random_state=0),
          DecisionTreeClassifier(max_depth=5),
          MLPClassifier(alpha=1, max_iter=1000),
          AdaBoostClassifier(),
          GaussianNB(),
          ]

def main(parser):
    args = parser.parse_args()
    model_dir = args.model_dir
    label_to_exclude = args.label_to_exclude
    config_path = args.config_path

    with open(os.getcwd() + config_path) as f:
        config = json.load(f)

    df_test, df_test_values = load_all_data_encoder(config["test_file"], label_to_exclude, config["label"], config["filter_label"], config["filter_label_value"], is_preprocess=True)
    df_test.drop(columns=['features'], inplace=True)

    labels = list(df_test['labels'].unique())
    labels_list = labels

    df_train, df_train_values = load_all_data_encoder(config["train_file"], label_to_exclude, config["label"],
                                              config["filter_label"], config["filter_label_value"], is_preprocess=True)
    df_train.drop(columns=['features'], inplace=True)
    labels = list(df_train['labels'].unique())
    tfidf = create_tf_idf_model(df_train['text'].values)
    df_train = convert_text_to_features(df_train, tfidf=tfidf, text_column_name='text')
    df_test = convert_text_to_features(df_test, tfidf=tfidf, text_column_name='text')
    if config["is_training"]:
        labels_list = labels
        wandb_config = {}
        df_eval = None
        y = df_train.iloc[:, 0].values
        X = df_train.iloc[:, 1:].values
        y_test = df_test.iloc[:, 0].values
        X_test = df_test.iloc[:, 1:].values
        for model in models:
            model.fit(X, y)
            print(str(model))
            print(model.score(X, y))
            y_pred = model.predict(X_test)
            result, f1 = scorePredict(y_test, y_pred, labels)
            print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--config_path",
                        default="/data/VIL_1/vil.json",
                        type=str,
                        help="File path to configuration parameters.")

    parser.add_argument("--model_dir",
                        default="",
                        type=str,
                        help="This parameter is the relative dir of model for predict.")

    parser.add_argument('--label_to_exclude',
                        default=[],
                        nargs='+',
                        help="This parameter should be used if you want to execute experiments with fewer classes.")

    main(parser)