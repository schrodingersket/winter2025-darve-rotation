from transformers import AutoModelForMaskedLM, AutoTokenizer
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from transformers import TrainingArguments, Trainer


# load PhysBERT
model_name = "thellert/physbert_cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

# load training data (PDG)
data_files = {"train": "pdg_text_nougat.txt"}
dataset = load_dataset("text", data_files=data_files)

# tokenize and mask
def tokenize_and_mask(examples):
    tokenized_inputs = tokenizer(
        examples["text"], 
        padding="max_length", 
        truncation=True, 
        max_length=512, 
        return_tensors="pt"
    )

    labels = tokenized_inputs["input_ids"].clone() 

    batch_size, seq_length = labels.shape

    probability_matrix = torch.full(labels.shape, 0.15)  # 15% probability per token

    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(seq.tolist(), already_has_special_tokens=True) 
        for seq in labels
    ]
    special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
    
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100

    # Replace masked tokens with [MASK]
    tokenized_inputs["input_ids"][masked_indices] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_dataset = dataset.map(tokenize_and_mask, batched=True)
train_dataloader = DataLoader(tokenized_dataset["train"], batch_size=8, shuffle=True)

# fine tune the model
training_args = TrainingArguments(
    output_dir="./physbert_mlm",
    evaluation_strategy="no",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    learning_rate=1e-4,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=500,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
)

trainer.train()

# save the fine tuned model
model.save_pretrained("./physbert_finetuned")
tokenizer.save_pretrained("./physbert_finetuned")