import json
from pathlib import Path

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model


# ============================================================
# PROJECT PATH
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

DATASET_NAME = "legal_training_data_expanded.jsonl"

matches = list(PROJECT_DIR.rglob(DATASET_NAME))

if not matches:
    raise FileNotFoundError(
        "legal_training_data_expanded.jsonl not found."
    )

DATA_FILE = matches[0]


# ============================================================
# BASE MODEL
# ============================================================

BASE_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"

# NEW MODEL - V3
OUTPUT_DIR = PROJECT_DIR / "models" / "your_legal_ai_v3"


# Safety checks
if "Qwen" in BASE_MODEL:
    raise RuntimeError("Qwen is not being used.")

if "1.7B" in BASE_MODEL:
    raise RuntimeError("Wrong model selected.")


print()
print("=" * 65)
print("           YOUR LEGAL AI - V3 TRAINING")
print("=" * 65)

print()
print("Dataset :", DATA_FILE)
print("Model   :", BASE_MODEL)
print("Output  :", OUTPUT_DIR)
print()


# ============================================================
# LOAD DATASET
# ============================================================

print("[1/6] Loading legal dataset...")

records = []

with open(DATA_FILE, "r", encoding="utf-8") as f:

    for line in f:

        if line.strip():
            records.append(json.loads(line))


print("Training examples:", len(records))

if len(records) == 0:
    raise ValueError("Dataset is empty.")


# ============================================================
# CHECK DATASET
# ============================================================

required_fields = [
    "instruction",
    "input",
    "output",
]

for number, record in enumerate(records, start=1):

    for field in required_fields:

        if field not in record:
            raise ValueError(
                f"Missing '{field}' in example {number}"
            )


print("Dataset format: OK")


# ============================================================
# TOKENIZER
# ============================================================

print()
print("[2/6] Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ============================================================
# PREPARE CHAT DATA
# ============================================================

print()
print("[3/6] Preparing clean instruction-response data...")


def prepare_example(example):

    instruction = str(
        example["instruction"]
    ).strip()

    input_text = str(
        example["input"]
    ).strip()

    answer = str(
        example["output"]
    ).strip()


    user_message = (
        instruction
        + "\n\n"
        + input_text
    )


    messages = [
        {
            "role": "user",
            "content": user_message,
        },
        {
            "role": "assistant",
            "content": answer,
        },
    ]


    prompt_messages = [
        {
            "role": "user",
            "content": user_message,
        },
    ]


    # Full conversation
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


    # Prompt only
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )


    # Tokenize full conversation
    full = tokenizer(
        full_text,
        truncation=True,
        max_length=256,
        padding="max_length",
    )


    # Tokenize prompt
    prompt = tokenizer(
        prompt_text,
        truncation=True,
        max_length=256,
        padding=False,
    )


    input_ids = full["input_ids"]
    attention_mask = full["attention_mask"]

    labels = input_ids.copy()


    # --------------------------------------------------------
    # Ignore prompt tokens
    # Model learns mainly the answer
    # --------------------------------------------------------

    prompt_length = len(
        prompt["input_ids"]
    )

    prompt_length = min(
        prompt_length,
        len(labels)
    )


    for i in range(prompt_length):
        labels[i] = -100


    # Ignore padding tokens
    for i in range(len(labels)):

        if attention_mask[i] == 0:
            labels[i] = -100


    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


dataset = Dataset.from_list(records)

tokenized_dataset = dataset.map(
    prepare_example,
    remove_columns=dataset.column_names,
)


print(
    "Clean response-focused dataset ready."
)


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("[4/6] Loading SmolLM2-360M...")

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL
)

print("SmolLM2-360M loaded successfully.")


# ============================================================
# LoRA
# ============================================================

print()
print("[5/6] Configuring LoRA...")


lora_config = LoraConfig(

    r=4,

    lora_alpha=8,

    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],

    lora_dropout=0.05,

    bias="none",

    task_type="CAUSAL_LM",
)


model = get_peft_model(
    model,
    lora_config
)


print()
print("LoRA enabled.")

model.print_trainable_parameters()


# ============================================================
# TRAINING SETTINGS
# ============================================================

training_args = TrainingArguments(

    output_dir=str(OUTPUT_DIR),

    # CPU friendly
    num_train_epochs=1,

    per_device_train_batch_size=1,

    gradient_accumulation_steps=1,

    learning_rate=1e-4,

    logging_steps=1,

    save_strategy="epoch",

    save_total_limit=1,

    report_to="none",

    fp16=False,

    dataloader_pin_memory=False,

    remove_unused_columns=False,
)


# ============================================================
# TRAINER
# ============================================================

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=tokenized_dataset,
)


# ============================================================
# START TRAINING
# ============================================================

print()
print("=" * 65)
print("           STARTING LEGAL AI V3")
print("=" * 65)

print()
print("Base Model : SmolLM2-360M-Instruct")
print("Examples   :", len(tokenized_dataset))
print("Epochs     : 1")
print("LoRA       : Enabled")
print("Training   : Response-focused")
print("Version    : V3")
print()

trainer.train()


# ============================================================
# SAVE
# ============================================================

print()
print("=" * 65)
print("           SAVING LEGAL AI V3")
print("=" * 65)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


model.save_pretrained(
    OUTPUT_DIR
)

tokenizer.save_pretrained(
    OUTPUT_DIR
)


print()
print("=" * 65)
print("           LEGAL AI V3 COMPLETE")
print("=" * 65)

print()
print("Saved at:")
print(OUTPUT_DIR)

print()