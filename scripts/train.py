import argparse
import logging
import os
import sys

import numpy as np
import torch
from datasets import load_from_disk, load_metric
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=32)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=64)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--model_id", type=str)
    parser.add_argument("--learning_rate", type=str, default=5e-5)
    parser.add_argument("--fp16", type=bool, default=True)

    # Data, model, and output directories
    parser.add_argument("--output_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--training_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--eval_dir", type=str, default=os.environ["SM_CHANNEL_EVAL"])

    # logging & evaluation strategie
    parser.add_argument("--logging_dir", type=str, default=os.path.join(os.environ["SM_MODEL_DIR"], "logs"))
    parser.add_argument("--strategy", type=str, default="steps")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--load_best_model_at_end", type=bool, default=True)
    parser.add_argument("--metric_for_best_model", type=str, default="f1")
    parser.add_argument("--report_to", type=str, default="tensorboard")

    # Push to Hub Parameters
    parser.add_argument("--push_to_hub", type=bool, default=True)
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument("--hub_strategy", type=str, default="every_save")
    parser.add_argument("--hub_token", type=str, default=None)

    args, _ = parser.parse_known_args()

    # Set up logging
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # load datasets
    train_dataset = load_from_disk(args.training_dir)
    eval_dataset = load_from_disk(args.eval_dir)

    logger.info(f" loaded train_dataset length is: {len(train_dataset)}")
    logger.info(f" loaded test_dataset length is: {len(eval_dataset)}")

    # define metrics and metrics function
    f1_metric = load_metric("f1")
    accuracy_metric = load_metric("accuracy")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        acc = accuracy_metric.compute(predictions=predictions, references=labels)
        f1 = f1_metric.compute(predictions=predictions, references=labels, average="micro")
        return {
            "accuracy": acc["accuracy"],
            "f1": f1["f1"],
        }

    # Prepare model labels - useful in inference API
    labels = train_dataset.features["labels"].names
    num_labels = len(labels)
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    # download model from model hub
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_id, num_labels=num_labels, label2id=label2id, id2label=id2label
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    # define training args
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        warmup_steps=args.warmup_steps,
        fp16=args.fp16,
        learning_rate=float(args.learning_rate),
        seed=33,
        # logging & evaluation strategies
        logging_dir=args.logging_dir,
        logging_strategy=args.strategy,  # to get more information to TB
        logging_steps=args.steps,  # to get more information to TB
        evaluation_strategy=args.strategy,
        eval_steps=args.steps,
        save_strategy=args.strategy,
        save_steps=args.steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=args.metric_for_best_model,
        report_to=args.report_to,
        # push to hub parameters
        push_to_hub=args.push_to_hub,
        hub_strategy="every_save",
        hub_model_id=args.hub_model_id,
        hub_token=args.hub_token,
    )

    # create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    # train model
    trainer.train()

    # evaluate model
    eval_result = trainer.evaluate(eval_dataset=eval_dataset)

    # save best model, metrics and create model card
    trainer.create_model_card(model_name=args.hub_model_id)
    trainer.push_to_hub()

    # Saves the model to s3 uses os.environ["SM_MODEL_DIR"] to make sure checkpointing works
    trainer.save_model(os.environ["SM_MODEL_DIR"])
