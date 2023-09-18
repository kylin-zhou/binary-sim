from tokenizers import Tokenizer, normalizers, pre_tokenizers
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

import os
from transformers import (
    BigBirdConfig,
    BigBirdForMaskedLM,
    LineByLineTextDataset,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
import torch
from torch.utils.data.dataset import Dataset, Iterable
from pathlib import Path
import linecache


def get_tokenizer(
    corpus="data/raw.txt",
    tokenizer_file="checkpoint/ByteBERT",
):
    """
    https://itcn.blog/p/20041134851.html
    https://github.com/huggingface/tokenizers/issues/325
    """
    if os.path.exists(tokenizer_file):
        # tokenizer = Tokenizer.from_file(tokenizer_file)
        tokenizer = PreTrainedTokenizerFast(
            model_max_length=1024, tokenizer_file=tokenizer_file
        )
        tokenizer.add_special_tokens(
            {
                "bos_token": "[CLS]",
                "eos_token": "[SEP]",
                "sep_token": "[SEP]",
                "cls_token": "[CLS]",
                "pad_token": "[PAD]",
                "mask_token": "[MASK]",
                "unk_token": "[UNK]",
            }
        )
    else:
        # init
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))

        # normalizer
        normalizer = normalizers.Sequence([normalizers.Lowercase()])
        tokenizer.normalizer = normalizer

        # tokenizer
        tokenizer.pre_tokenizer = Whitespace()

        # trainer
        trainer = WordLevelTrainer(
            vocab_size=261,
            special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
        )
        tokenizer.train(files=[corpus], trainer=trainer)

        # post_processor
        tokenizer.post_processor = TemplateProcessing(
            single="[CLS] $A [SEP]",
            special_tokens=[
                ("[CLS]", tokenizer.token_to_id("[CLS]")),
                ("[SEP]", tokenizer.token_to_id("[SEP]")),
            ],
        )

        # save
        tokenizer.save(tokenizer_file)

        tokenizer = PreTrainedTokenizerFast(
            model_max_length=1024, tokenizer_file=tokenizer_file
        )
        tokenizer.add_special_tokens(
            {
                "bos_token": "[CLS]",
                "eos_token": "[SEP]",
                "sep_token": "[SEP]",
                "cls_token": "[CLS]",
                "pad_token": "[PAD]",
                "mask_token": "[MASK]",
                "unk_token": "[UNK]",
            }
        )

    return tokenizer


class ByteDataset(Dataset):
    def __init__(self, tokenizer, file_path, maxlen=1024, evaluate: bool = False):
        self.tokenizer = tokenizer
        self._filename = file_path
        self.maxlen = maxlen

    def __getitem__(self, idx):
        line = linecache.getline(self._filename, idx + 1)
        encoding = self.tokenizer(
            line, max_length=self.maxlen, truncation=True, padding=True
        )["input_ids"]
        example = {"input_ids": torch.tensor(encoding, dtype=torch.long)}
        return example

    def __len__(self):
        return 1


def get_parnum(m):
    total = sum([param.nelement() for param in m.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))


def train():
    config = BigBirdConfig(
        vocab_size=261,
        hidden_size=128,
        num_attention_heads=8,
        num_hidden_layers=4,
        max_position_embeddings=1024,
    )
    model = BigBirdForMaskedLM(config=config)
    get_parnum(model)
    tokenizer = get_tokenizer()

    print("开始读取dataset")
    dataset = ByteDataset(
        tokenizer=tokenizer,
        file_path="data/raw.txt",
        maxlen=1024,
    )
    print("开始读取data_collator")
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
    )

    training_args = TrainingArguments(
        output_dir="checkpoint/BigBirdData_bytecode_lm1024_Checkpoint",
        overwrite_output_dir=True,
        num_train_epochs=40,
        per_device_train_batch_size=128,
        save_steps=1000,
        save_total_limit=500,
        prediction_loss_only=True,
        gradient_checkpointing=True,
        gradient_accumulation_steps=4,  # 累计多个batch，更新一次参数
        max_grad_norm=1,
        group_by_length=True,  # 相似长度sample放在一起
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )
    print("开始训练")
    trainer.train()
    trainer.save_model(
        "checkpoint/BigBirdData_bytecode_lm1024_Output"
    )


if __name__ == "__main__":
    train()
