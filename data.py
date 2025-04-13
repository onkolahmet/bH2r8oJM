from datasets import load_dataset



def get_dataset(dataset_id = "gretelai/synthetic_text_to_sql", split = "train[:10000]"):
    dataset = load_dataset(dataset_id, split=split)
    return dataset

def prepare_data(examples):
    """
    Formats a batch of examples for the SFTTrainer.
    Returns a list of strings with prompts and responses.
    """
    formatted_texts = []
    
    # Process each example in the batch
    for i in range(len(examples['sql_prompt'])):
        # Initialize prompt
        prompt = "Translate natural language questions to SQL queries.\n\n"
        
        # Add context if available
        if 'sql_context' in examples and examples['sql_context'][i]:
            prompt += f"Table Context:\n{examples['sql_context'][i]}\n\n"
        
        # Add the question
        prompt += f"Q: {examples['sql_prompt'][i]}\nSQL:"
        
        # Add the response (will be masked by the collator)
        full_text = f"{prompt} ### The response query is: {examples['sql'][i]}"
        
        formatted_texts.append(full_text)
    
    return formatted_texts