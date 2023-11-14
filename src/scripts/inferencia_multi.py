import argparse
import os
import pickle
import json
import torch
from common.utils import load_all_data
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix,classification_report
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from torch.utils.data import DataLoader, SequentialSampler


def main(args):
    config_path = args.config_path

    with open(os.getcwd() + config_path) as f:
        config = json.load(f)

    df_test = load_all_data(config["test_file"], config["label"])
    peft_model_id = os.path.join(args.adapter_type, args.experiment, "assets")

    peft_config = PeftConfig.from_pretrained(peft_model_id)

    # load base LLM model and tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(peft_config.base_model_name_or_path
        # load_in_8bit=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)
    # Load the Lora model
    model = PeftModel.from_pretrained(model, peft_model_id)
    if config["use_cuda"]:
         model.cuda()
    model.eval()

    predictions, references, text_input = [], [], []

    for idx, row in tqdm(df_test.iterrows()):
        text_input.append("Clasifica el siguiente tuit en una categoría: "
        + row["text"].replace("\n", " ")
        + " La respuesta es: ")

    input_ids = tokenizer(text_input, return_tensors="pt", truncation=True, is_split_into_words=False, padding=True).input_ids.cuda()
    eval_sampler = SequentialSampler(input_ids)
    eval_dataloader = DataLoader(
        input_ids, sampler=eval_sampler, batch_size=32
    )
    with torch.inference_mode():
        for i, batch in enumerate(tqdm(eval_dataloader, desc="Running Evaluation")):
            outputs = model.generate(input_ids=batch, do_sample=True, top_p=0.9, max_new_tokens=20, temperature=1e-3)
            predictions.extend(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True))
    references = df_test["label"].tolist()

    metrics = {}
    metrics["accuracy"] = accuracy_score(references, predictions)
    metrics["f1-macro"] = f1_score(references, predictions, average="macro")
    metrics["f1-weighted"] = f1_score(references, predictions, average="weighted")
    print(metrics)
    print(confusion_matrix(references, predictions, labels=list(set(references))))
    print(classification_report(references, predictions, digits=5))

    metrics_dir = os.path.join(args.adapter_type, args.experiment, "metrics")
    if not os.path.exists(metrics_dir):
        os.makedirs(metrics_dir)
    with open(os.path.join(metrics_dir, "metric.pkl"), "wb") as handle:
        pickle.dump(metrics, handle)
    print(f"Inference over for {args.experiment}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_type", default="experiments")
    parser.add_argument(
        "--experiment", default="classification_lora_samples-4750_epochs-3_r-16_dropout-0.1_model-google/flan-t5-large"
    )
    parser.add_argument("--config_path", default="/data/VIL_1/vil.json")
    args = parser.parse_args()

    main(args)