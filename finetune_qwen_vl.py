import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model
import os
from PIL import Image

# --- Configuration ---
MODEL_ID = "Qwen/Qwen-VL-Chat" # Revert to original model
DATASET_PATH = "." # Directory containing train.json and val.json
OUTPUT_DIR = "./qwen_vl_khatt_finetuned"
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
# Adjust target modules based on Qwen-VL architecture if needed
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] # Common for LLMs

# Training Arguments (Adjust based on your hardware and needs)
PER_DEVICE_TRAIN_BATCH_SIZE = 1 # Reduce drastically for memory
PER_DEVICE_EVAL_BATCH_SIZE = 1  # Reduce drastically for memory
GRADIENT_ACCUMULATION_STEPS = 16 # Increase to compensate batch size reduction
LEARNING_RATE = 1e-4
NUM_TRAIN_EPOCHS = 1 # Start with 1 epoch, increase if needed
LOGGING_STEPS = 10
EVAL_STEPS = 50 # Evaluate periodically
SAVE_STEPS = 100
FP16 = True # Use mixed precision if supported
OPTIM = "adamw_torch"

# --- Load Dataset ---
print("Loading dataset...")
raw_datasets = load_dataset('json', data_files={'train': os.path.join(DATASET_PATH, 'train.json'),
                                                'validation': os.path.join(DATASET_PATH, 'val.json')})
print(f"Dataset loaded: {raw_datasets}")

# --- Load Model and Processor ---
print(f"Loading model and processor: {MODEL_ID}...")

# Re-introduce Quantization config for memory efficiency
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

# Load processor (handles text tokenization and image processing)
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

# Load model with quantization
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=quantization_config, # Use 4-bit quantization
    trust_remote_code=True,
    device_map="auto" # Automatically distribute model across available GPUs
)
print("Model and processor loaded.")

# --- Preprocessing Function ---
def preprocess_data(examples):
    """Prepares image and text data for Qwen-VL."""
    image_paths = examples['image']
    prompts = examples['prompt']
    transcriptions = examples['transcription']

    batch_images = []
    batch_texts = []

    for img_path, prompt, transcription in zip(image_paths, prompts, transcriptions):
        try:
            # Load image
            image = Image.open(img_path).convert('RGB')
            batch_images.append(image)

            # Format text input for Qwen-VL Chat (adjust if using a non-chat model)
            # Example format: "<img>path/to/image.jpg</img>\n{prompt}\nAnswer:{transcription}"
            # The processor usually handles the image token insertion based on text structure.
            # We need to construct the text part that includes the prompt and the expected answer.
            # Qwen-VL format might involve specific roles or tags. Consulting model card is best.
            # Simple approach: Combine prompt and transcription for training label.
            # The processor needs text input structured correctly. Let's try a basic query format.
            query = processor.tokenizer.from_list_format([
                {'image': img_path}, # The processor should convert this path to an image token
                {'text': f"{prompt}\nAnswer:"}
            ])
            # The label should be the expected transcription
            label_text = transcription + processor.tokenizer.eos_token # Add EOS token to label

            batch_texts.append(query)

        except FileNotFoundError:
            print(f"Warning: Image file not found: {img_path}. Skipping.")
            # Need to handle this skip properly if using map, maybe return None or filter later
            continue
        except Exception as e:
            print(f"Error processing image {img_path}: {e}. Skipping.")
            continue

    # Process batch using the processor
    # Padding=True might be needed, or use DataCollator
    inputs = processor(text=batch_texts, images=batch_images, return_tensors="pt", padding="longest")

    # Tokenize labels separately (ensure padding/truncation matches if needed)
    # For Causal LM, labels are typically the input_ids shifted.
    # The Trainer usually handles this if labels aren't provided explicitly.
    # Let's prepare labels explicitly for clarity.
    labels = processor.tokenizer(text=transcriptions, return_tensors="pt", padding="longest", truncation=True).input_ids
    # Replace padding token id in labels with -100 to ignore loss calculation
    labels[labels == processor.tokenizer.pad_token_id] = -100
    inputs['labels'] = labels

    return inputs


print("Preprocessing datasets...")
# Apply preprocessing (might need adjustments based on how processor handles batches)
# Consider using `batched=True` and adjusting the function if performance is an issue.
# For simplicity, let's try without batching first, though it will be slow.
# Update: Using batched=True requires the function to handle lists of inputs.
tokenized_datasets = raw_datasets.map(preprocess_data, batched=True, remove_columns=raw_datasets["train"].column_names)
print(f"Preprocessing complete. Tokenized datasets: {tokenized_datasets}")


# --- Configure LoRA ---
print("Configuring LoRA...")
lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    target_modules=LORA_TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM" # Important for Causal LM tasks
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
print("LoRA configured.")

# --- Training Arguments ---
print("Setting up Training Arguments...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_strategy="steps",
    logging_steps=LOGGING_STEPS,
    evaluation_strategy="steps",
    eval_steps=EVAL_STEPS,
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=2, # Keep only the last 2 checkpoints
    fp16=FP16,
    optim=OPTIM,
    dataloader_num_workers=4, # Adjust based on your system
    # report_to="tensorboard", # Optional: if you want to use TensorBoard
    load_best_model_at_end=True, # Optional: load the best model based on eval loss
    metric_for_best_model="eval_loss", # Optional
    greater_is_better=False, # Optional: for loss, lower is better
    remove_unused_columns=False, # Important when using custom preprocessing
    trust_remote_code=True,
)
print("Training Arguments set.")

# --- Initialize Trainer ---
print("Initializing Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    processor=processor, # Pass processor for potential use in data collation/saving
    # Data collator might be needed if padding isn't handled perfectly by map/processor
    # data_collator=DataCollatorForLanguageModeling(tokenizer=processor.tokenizer, mlm=False) # Example
)
print("Trainer initialized.")

# --- Start Training ---
print("Starting training...")
train_result = trainer.train()

# --- Save Final Model ---
print("Training finished. Saving final LoRA adapter...")
# Saves the LoRA adapter weights, not the full model
trainer.save_model(os.path.join(OUTPUT_DIR, "final_adapter"))
# You might also want to save the processor
processor.save_pretrained(os.path.join(OUTPUT_DIR, "final_adapter"))

print(f"Training complete. Results: {train_result}")
print(f"Final LoRA adapter saved to: {os.path.join(OUTPUT_DIR, 'final_adapter')}")
