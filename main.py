import warnings
warnings.filterwarnings("ignore")

import argparse
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import TrainingArguments
from get_model import get_qwen_model
from data import get_dataset, prepare_data
from huggingface_hub import login
import time

def main(args):
    tokenizer, model = get_qwen_model(model_id = args.model_id)
    data = get_dataset(dataset_id=args.dataset_id)
    
    training_params = TrainingArguments(
    output_dir="./model",
    save_strategy="epoch",  # Changed from "steps" to "epoch"
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    logging_steps=500,
    learning_rate=1e-4,
    push_to_hub=True,
    hub_model_id=f"{args.model_id.split('/')[1]}-SQL-generator",
    push_to_hub_model_id=f"{args.model_id.split('/')[1]}-SQL-generator",
    save_total_limit=2,
    fp16=True,
    gradient_accumulation_steps=4 
)
    
    response_template = " ### The response query is:"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
    
    trainer = SFTTrainer(
    model,
    args=training_params,
    train_dataset=data,
    formatting_func=prepare_data,
    data_collator=collator,
    )
    t1 = time.time()
    trainer.train()
    print("TIME TAKEN: ", time.time() - t1)
    trainer.save_model(f"./{args.model_id.split('/')[1]}")
    
    trainer.push_to_hub(f"{args.user_name/args.model_id.split('/')[1]}-SQL-generator")
    
    
if __name__ == "__main__":
    # model_id, dataset_id, epochs, batch_size
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default='Qwen/Qwen2-0.5B-Instruct')
    parser.add_argument('--dataset_id', type=str, default='gretelai/synthetic_text_to_sql')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--hf_login', type=bool, default=True)
    parser.add_argument('--user_name', type=str, default='your_username')
    args = parser.parse_args()
    if args.hf_login:
        login()
    else:
        pass
    main(args)