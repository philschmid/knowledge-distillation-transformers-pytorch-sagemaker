# Task-specific knowledge distillation with BERT, Transformers & Amazon SageMaker

Welcome to our end-to-end task-specific knowledge distilattion Text-Classification example using Transformers, PyTorch & Amazon SageMaker. Distillation is the process of training a small "student" to mimic a larger "teacher". In this example, we will use [BERT-base](https://huggingface.co/textattack/bert-base-uncased-SST-2) as Teacher and [https://huggingface.co/google/bert_uncased_L-2_H-128_A-2) as Student. We will use [Text-Classification](https://huggingface.co/tasks/text-classification) as task-specific knowledge distillation task and the [Stanford Sentiment Treebank v2 (SST-2)](https://paperswithcode.com/dataset/sst) dataset for training.

This Repository contains to Notebooks: 

* [knowledge-distillation](knowledge-distillation.ipynb) a step-by-step example on how distil the knowledge from the teacher to the student.
* [sagemaker-distillation](sagemaker-distillation.ipynb) a derived version of the first notebook, which shows how to scale your training for distributed training using Amazon SageMaker.


https://huggingface.co/M-FAC/bert-tiny-finetuned-sst2