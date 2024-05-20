import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.file_download")
from transformers import GPTNeoForCausalLM, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import torch
from prepare_data import data

# Load the tokenizer and model
model_name = "EleutherAI/gpt-neo-2.7B"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Add a padding token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

model = GPTNeoForCausalLM.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))  # Resize the token embeddings if new tokens are added

# Prepare the dataset for training
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.input_ids = []
        self.attn_masks = []
        self.labels = []

        for text in texts:
            if isinstance(text, str):
                encodings_dict = tokenizer(text, truncation=True, max_length=max_length, padding="max_length")
                self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
                self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))
                self.labels.append(torch.tensor(encodings_dict['input_ids']))
            else:
                raise ValueError("Each text input must be of type `str`")

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attn_masks[idx],
            'labels': self.labels[idx]
        }

# Create the dataset
train_dataset = TextDataset(data, tokenizer)

# Define data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=1,  # Lower batch size for testing
    per_device_eval_batch_size=1,   # Lower batch size for testing
    warmup_steps=10,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
)

# Create the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
)

# Train the model
trainer.train()

# Create a pipeline for text generation
from transformers import pipeline

text_gen_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Example usage
prompt = "The future of AI in the food and beverage industry is"
output = text_gen_pipeline(prompt, max_length=100, num_return_sequences=1)
print(output)