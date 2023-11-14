from typing import List, Union
import pandas as pd
from datasets import load_dataset, concatenate_datasets, DatasetDict
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
import datasets
import os
import re
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def de_emojify(text):
    regrex_pattern = re.compile(pattern="["
                                        u"\U0001F600-\U0001F92F"  # emoticons
                                        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                        u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                        u"\U00002702-\U000027B0"
                                        u"\U000024C2-\U0001F251"
                                        u"\U0001F190-\U0001F1FF"
                                        u"\U0001F926-\U0001FA9F"
                                        u"\u2640-\u2642"
                                        u"\u2600-\u2B55"
                                        u"\u200d"
                                        u"\u23cf"
                                        u"\u23e9"
                                        u"\u231a"
                                        u"\ufe0f"
                                        "]+", flags=re.UNICODE)
    return regrex_pattern.sub(r'', text)


def preprocess(value):
    new_value = de_emojify(value)
    new_value = re.sub(r'http\S+', '', new_value)
    return new_value


def load_all_data(file_in, label, labels_to_exclude=[], filter_label=None, filter_label_value=None, is_preprocess=False):
    if file_in.endswith('.tsv'):
        df_in = pd.read_csv(os.getcwd() + file_in, sep='\t')
    else:
        df_in = pd.read_json(os.getcwd() + file_in, lines=True)

    for value in labels_to_exclude:
        df_in = df_in[df_in[label] != value]
    if label in df_in.columns:
        if filter_label:
            df_in = df_in[df_in[filter_label] == filter_label_value]
        print(df_in[label].value_counts())
        labels = df_in[label]
    if is_preprocess:
        df_in['text'] = df_in['text'].apply(preprocess)
    list_of_tuples = list(zip(list(df_in['text']), list(labels)))
    df = pd.DataFrame(list_of_tuples, columns=['text', 'label'])
    return df


def load_all_data_encoder(file_in, labels_to_exclude, label, filter_label, filter_label_value, is_preprocess=False):
    if file_in.endswith('.tsv'):
        df_in = pd.read_csv(os.getcwd() + file_in, sep='\t')
    else:
        df_in = pd.read_json(os.getcwd() + file_in, lines=True)

    for value in labels_to_exclude:
        df_in = df_in[df_in[label] != value]
    if label in df_in.columns:
        if filter_label:
            df_in = df_in[df_in[filter_label] == filter_label_value]
        print(df_in[label].value_counts())
        labels = df_in[label]
        # To labels
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(labels.values)
    features = [[]] * len(df_in)
    if is_preprocess:
        df_in['text'] = df_in['text'].apply(preprocess)
    list_of_tuples = list(zip(list(df_in['text']), list(labels), features))
    df = pd.DataFrame(list_of_tuples, columns=['text', 'labels', 'features'])
    return df, df_in

def get_data_format(
    args, tokenizer: AutoTokenizer, train_df: pd.DataFrame
) -> List[Union[DatasetDict, int, int]]:
    eval_df = None
    # sample n points from train_df
    train_df, eval_df = train_test_split(
        train_df,
        train_size=args.train_sample_fraction,
        stratify=train_df["label"],
    )

    train_dataset = datasets.Dataset.from_pandas(train_df)
    eval_dataset = datasets.Dataset.from_pandas(eval_df)

    dataset = DatasetDict({"train": train_dataset, "eval": eval_dataset})

    print(f"Train dataset size: {len(dataset['train'])}")
    print(f"Eval dataset size: {len(dataset['eval'])}")

    tokenized_inputs = concatenate_datasets([dataset["train"], dataset["eval"]]).map(
        lambda x: tokenizer(x["text"], truncation=True),
        batched=True,
        remove_columns=["text", "label"],
    )

    max_source_length = max([len(x) for x in tokenized_inputs["input_ids"]])
    print(f"Max source length: {max_source_length}")

    tokenized_targets = concatenate_datasets([dataset["train"], dataset["eval"]]).map(
        lambda x: tokenizer(x["label"], truncation=True),
        batched=True,
        remove_columns=["text", "label"],
    )

    max_target_length = max([len(x) for x in tokenized_targets["input_ids"]])
    print(f"Max target length: {max_target_length}")

    return dataset, max_source_length, max_target_length


def parse_wandb_param(sweep_config, model_args):
    # Extracting the hyperparameter values
    cleaned_args = {}
    layer_params = []
    param_groups = []
    for key, value in sweep_config.items():
        if isinstance(value, dict):
            value = value[0]
        if key.startswith("layer_"):
            # These are layer parameters
            layer_keys = key.split("_")[-1]

            # Get the start and end layers
            start_layer = int(layer_keys.split("-")[0])
            end_layer = int(layer_keys.split("-")[-1])

            # Add each layer and its value to the list of layer parameters
            for layer_key in range(start_layer, end_layer):
                layer_params.append(
                    {"layer": layer_key, "lr": value,}
                )
        elif key.startswith("params_"):
            # These are parameter groups (classifier)
            params_key = key.split("_")[-1]
            param_groups.append(
                {
                    "params": [params_key],
                    "lr": value,
                    "weight_decay": model_args.weight_decay
                    if "bias" not in params_key
                    else 0.0,
                }
            )
        else:
            # Other hyperparameters (single value)
            cleaned_args[key] = value
    cleaned_args["custom_layer_parameters"] = layer_params
    cleaned_args["custom_parameter_groups"] = param_groups

    # Update the model_args with the extracted hyperparameter values
    model_args.update(cleaned_args)


def scorePredict(y_test, y_hat, labels):
    from sklearn.metrics import f1_score

    matriz = confusion_matrix(y_test, y_hat, labels=labels)
    f1_score_value = f1_score(y_test, y_hat, average='macro')
    value_class = classification_report(y_test, y_hat, digits=5)
    result = 'Matriz de Confusión:' + '\n' + str(matriz) + '\n' + value_class + '\n' + str(f1_score_value)
    return result, f1_score_value