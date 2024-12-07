# 2024 iFLYTEK A.I. 开发者大赛：人岗匹配挑战赛 赛季3 - 第4名方案

## 1. 赛事背景
讯飞智聘是一款面向企业招聘全流程的智能化解决方案，运用科大讯飞的智能语音、自然语言理解和计算机视觉技术，帮助企业提升招聘效率、降低成本。特别是在校园招聘等场景，面对大量简历，如何快速筛选出最适合的候选人，并将其与合适岗位匹配，是招聘过程中亟待解决的难题。

[赛题页面](https://challenge.xfyun.cn/topic/info?type=match-challenge-s3)

[数据集下载](https://challenge.xfyun.cn/topic/info?type=match-challenge-s3&option=stsj)

## 2. 赛事任务
本次大赛任务是构建一个智能人岗匹配系统，预测简历与岗位是否匹配。大赛提供了加密脱敏的数据集，其中包括大量的岗位JD和求职者简历，参赛者需要基于这些数据训练模型。

## 3. 方案概述
本方案利用BERT模型进行预训练和微调来完成人岗匹配任务。由于数据是加密的，每个字符映射为数字，不能直接使用开源模型，因此我们使用了自定义的预训练BERT模型，并进行微调优化。主要步骤包括：

1. 数据预处理：从简历中提取教育背景、工作经历等信息并拼接成句子。
2. 创建Tokenizer：自定义词汇表并基于BERTTokenizer进行训练。
3. 预训练BERT：使用Masked Language Model (MLM) 进行BERT模型的预训练。
4. 微调模型：使用BERT的分类头进行任务微调，进行简历与岗位的匹配预测。
5. 使用NSP模型进行实验，但未能成功优化。

## 4. 关键步骤及代码

### 4.1 数据预处理
首先，我们对简历中的教育背景数据进行预处理，将其转化为句子形式，方便BERT模型处理。

```python
def get_profileEduExps_sentence(profileEduExps_):
    """  
    根据学习阶段生成句子（如本科、研究生等）
    """
    profileEduExps_sentence = ["<begin_EduExp>", "<education>", profileEduExps_["education"].replace(" ", "&")] \
    + ["<schoolLevel>", " ".join(profileEduExps_["schoolLevel"]).replace(" ", "&")] \
    + ["<department>"] + profileEduExps_["department"].split() + ["<major>"] + profileEduExps_["major"].split() \
    + ["<courses>"] + profileEduExps_["courses"].split() + ["<school>"] + profileEduExps_["school"].split()  \
    + ["<GPA>"] + profileEduExps_["GPA"].split() + ["<ranking>"] + profileEduExps_["ranking"].split() \
    + ["<duration>"] + profileEduExps_["duration"].split() + ["<end_EduExp>"]
    return profileEduExps_sentence
```

### 4.2 创建Tokenizer
基于简历数据创建自定义词汇表，并使用BertTokenizer进行编码。

```python
from transformers import BertTokenizer

# 创建词汇表（去重）
all_tokens = set(token for sentence in corpus for token in sentence)
vocab = {token: idx for idx, token in enumerate(sorted(all_tokens), start=5)}

# 添加特殊 token (如: [PAD], [CLS], [SEP], [UNK])
special_tokens = {"[PAD]": 0, "[CLS]": 1, "[SEP]": 2, "[UNK]": 3, "[MASK]": 4}
vocab = {**special_tokens, **vocab}

# 保存词汇表
with open("../user_data/vocab.txt", "w", encoding="utf-8") as vocab_file:
    for token, idx in vocab.items():
        vocab_file.write(f"{token}\n")

# 加载词汇表并创建 BertTokenizer
tokenizer = BertTokenizer(vocab_file="../user_data/vocab.txt", do_lower_case=False)
```

### 4.3 预训练BERT模型
使用BERT的Masked Language Model进行预训练，生成自定义的BERT模型。

```python
from transformers import BertConfig, BertForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# 定义 BERT 模型配置
config = BertConfig(
    vocab_size=len(tokenizer.vocab),
    max_position_embeddings=dataset.max_length,
    hidden_size=768,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=2,
    pad_token_id=tokenizer.pad_token_id
)

# 初始化BERT模型
model = BertForMaskedLM(config=config)

# 数据收集器
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)

# 定义训练参数
training_args = TrainingArguments(
    output_dir="../user_data/output",
    num_train_epochs=100,
    per_device_train_batch_size=16,
    logging_strategy="epoch",
    save_strategy="epoch",
    logging_dir='../user_data/logs',
    fp16=True
)

# 训练模型
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
    tokenizer=tokenizer
)

trainer.train()
```

### 4.4 微调BERT模型
使用预训练的BERT模型进行微调，解决人岗匹配任务。

```python
from transformers import Trainer, TrainingArguments
from sklearn.utils import compute_class_weight
import torch

# 计算类别权重
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
class_weights = torch.tensor(class_weights, dtype=torch.float).to("cuda")

# 定义带权重的损失函数
loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

# 自定义Trainer
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels").to("cuda")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

# 微调模型
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_encoded,
    tokenizer=tokenizer
)
trainer.train()
```

### 4.5 NSP微调实验（未成功）
虽然尝试了基于BERT的NSP任务进行微调，但未能成功提升效果。

```python
from transformers import BertForNextSentencePrediction

# 加载BERT预训练模型权重
model = BertForNextSentencePrediction.from_pretrained(path)

# 构建NSP数据集并训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_encoded_nsp,
    eval_dataset=valid_encoded_nsp,
    tokenizer=tokenizer
)
trainer.train()
```

## 5. 总结
本方案通过使用BERT模型进行预训练和微调，结合自定义Tokenizer及带权重的损失函数，在加密数据的情况下有效地解决了人岗匹配问题。
