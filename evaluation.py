from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset
import evaluate
import pandas as pd
from tqdm import tqdm
import re
import sqlparse
import numpy as np


bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
chrf = evaluate.load("chrf")
exact_match = evaluate.load("exact_match")

try:
    squad = evaluate.load("squad")
except:
    squad = None

device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(
    "onkolahmet/Qwen2-0.5B-Instruct-SQL-generator", 
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("onkolahmet/Qwen2-0.5B-Instruct-SQL-generator")


dataset = load_dataset("gretelai/synthetic_text_to_sql")
test_data = dataset["test"]

# examples = [
#     {
#         "question": "Get the names and emails of customers who placed an order in the last 30 days.",
#         "sql": "SELECT name, email FROM customers WHERE order_date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY);"
#     },
#     {
#         "question": "Find all employees with a salary greater than 50000.",
#         "sql": "SELECT * FROM employees WHERE salary > 50000;"
#     },
#     {
#         "question": "List all product names and their categories where the price is below 50.",
#         "sql": "SELECT name, category FROM products WHERE price < 50;"
#     },
#     {
#         "question": "How many users registered in the year 2022?",
#         "sql": "SELECT COUNT(*) FROM users WHERE YEAR(registration_date) = 2022;"
#     }
# ]

def generate_sql(question, context=None):
    # Construct prompt with few-shot examples and context if available
    prompt = "Translate natural language questions to SQL queries.\n\n"
    
    # Add table context if available
    if context:
        prompt += f"Table Context:\n{context}\n\n"
    
        
    # Add few-shot examples
    # for ex in examples:
    #     prompt += f"Q: {ex['question']}\nSQL: {ex['sql']}\n\n"

    # Add the current question
    prompt += f"Q: {question}\nSQL:"
    
    # Tokenize and generate
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generate SQL query
    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=128,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id
    )
    
    # Extract and decode only the new generation
    sql_query = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
    return sql_query.strip()

def clean_sql_output(sql_text):
    """
    Clean and deduplicate SQL queries:
    1. Remove comments
    2. Remove duplicate queries
    3. Extract only the most relevant query
    4. Format properly
    """
    # Remove SQL comments (both single line and multi-line)
    sql_text = re.sub(r'--.*?$', '', sql_text, flags=re.MULTILINE)
    sql_text = re.sub(r'/\*.*?\*/', '', sql_text, flags=re.DOTALL)
    
    # Remove markdown code block syntax if present
    sql_text = re.sub(r'```sql|```', '', sql_text)
    
    # Split into individual queries if multiple exist
    if ';' in sql_text:
        queries = [q.strip() for q in sql_text.split(';') if q.strip()]
    else:
        # If no semicolons, try to identify separate queries by SELECT statements
        sql_text_cleaned = re.sub(r'\s+', ' ', sql_text)
        select_matches = list(re.finditer(r'SELECT\s+', sql_text_cleaned, re.IGNORECASE))
        
        if len(select_matches) > 1:
            queries = []
            for i in range(len(select_matches)):
                start = select_matches[i].start()
                end = select_matches[i+1].start() if i < len(select_matches) - 1 else len(sql_text_cleaned)
                queries.append(sql_text_cleaned[start:end].strip())
        else:
            queries = [sql_text]
    
    # Remove empty queries
    queries = [q for q in queries if q.strip()]
    
    if not queries:
        return ""
    
    # If we have multiple queries, need to deduplicate
    if len(queries) > 1:
        # Normalize queries for comparison (lowercase, remove extra spaces)
        normalized_queries = []
        for q in queries:
            # Use sqlparse to format and normalize
            try:
                formatted = sqlparse.format(
                    q + ('' if q.strip().endswith(';') else ';'), 
                    keyword_case='lower',
                    identifier_case='lower', 
                    strip_comments=True,
                    reindent=True
                )
                normalized_queries.append(formatted)
            except:
                # If sqlparse fails, just do basic normalization
                normalized = re.sub(r'\s+', ' ', q.lower().strip())
                normalized_queries.append(normalized)
        
        # Find unique queries
        unique_queries = []
        unique_normalized = []
        
        for i, norm_q in enumerate(normalized_queries):
            if norm_q not in unique_normalized:
                unique_normalized.append(norm_q)
                unique_queries.append(queries[i])
        
        # Choose the most likely correct query:
        # 1. Prefer queries with SELECT
        # 2. Prefer longer queries (often more detailed)
        # 3. Prefer first query if all else equal
        select_queries = [q for q in unique_queries if re.search(r'SELECT\s+', q, re.IGNORECASE)]
        
        if select_queries:
            # Choose the longest SELECT query (likely most detailed)
            best_query = max(select_queries, key=len)
        elif unique_queries:
            # If no SELECT queries, choose the longest query
            best_query = max(unique_queries, key=len)
        else:
            # Fallback to the first query
            best_query = queries[0]
    else:
        best_query = queries[0]
    
    # Clean up the chosen query
    best_query = best_query.strip()
    if not best_query.endswith(';'):
        best_query += ';'
    
    # Final formatting to ensure consistent spacing
    best_query = re.sub(r'\s+', ' ', best_query)
    
    return best_query

# New functions for SQL-specific evaluation

def normalize_sql_for_comparison(query):
    """Normalize SQL for better comparison"""
    try:
        # Remove trailing semicolons for uniform comparison
        query = query.strip()
        if query.endswith(';'):
            query = query[:-1]
        
        # Format with sqlparse for consistency
        formatted = sqlparse.format(
            query,
            keyword_case='upper',  # Standardize SQL keywords to uppercase
            identifier_case='lower',  # Standardize identifiers to lowercase
            strip_comments=True,
            reindent=True
        )
        
        # Additional normalization:
        # - Standardize whitespace
        formatted = re.sub(r'\s+', ' ', formatted)
        
        return formatted.strip()
    except:
        # If formatting fails, return the original with minimal normalization
        return re.sub(r'\s+', ' ', query.strip())

def calculate_sql_component_match(pred_query, ref_query):
    """
    Calculate structural match score based on SQL components
    Returns a score between 0 and 1 based on matching SQL components
    """
    try:
        # Normalize both queries
        pred_norm = normalize_sql_for_comparison(pred_query)
        ref_norm = normalize_sql_for_comparison(ref_query)
        
        # Extract key SQL components
        components = {
            'SELECT': re.compile(r'SELECT\s+(.*?)\s+FROM', re.IGNORECASE),
            'FROM': re.compile(r'FROM\s+(.*?)(?:\s+WHERE|\s+GROUP|\s+ORDER|\s+LIMIT|$)', re.IGNORECASE),
            'WHERE': re.compile(r'WHERE\s+(.*?)(?:\s+GROUP|\s+ORDER|\s+LIMIT|$)', re.IGNORECASE),
            'GROUP BY': re.compile(r'GROUP BY\s+(.*?)(?:\s+HAVING|\s+ORDER|\s+LIMIT|$)', re.IGNORECASE),
            'ORDER BY': re.compile(r'ORDER BY\s+(.*?)(?:\s+LIMIT|$)', re.IGNORECASE),
            'LIMIT': re.compile(r'LIMIT\s+(.*?)$', re.IGNORECASE)
        }
        
        # Extract components from both queries
        pred_components = {}
        ref_components = {}
        
        for name, pattern in components.items():
            pred_match = pattern.search(pred_norm)
            ref_match = pattern.search(ref_norm)
            
            pred_components[name] = pred_match.group(1).strip() if pred_match else None
            ref_components[name] = ref_match.group(1).strip() if ref_match else None
        
        # Calculate component matches
        matched_components = 0
        total_components = 0
        
        for name in components:
            # If reference has this component
            if ref_components[name]:
                total_components += 1
                # Check if prediction matches
                if pred_components[name] and pred_components[name].lower() == ref_components[name].lower():
                    matched_components += 1
        
        # If no components were found in reference, return 0
        if total_components == 0:
            return 0
        
        return matched_components / total_components
    
    except Exception as e:
        print(f"Error in component matching: {e}")
        return 0

def contains_key_entities(pred_query, ref_query):
    """Check if the prediction contains key entities from the reference query"""
    try:
        # Extract table names and column names from reference
        ref_tables = re.findall(r'FROM\s+([a-zA-Z0-9_]+)', ref_query, re.IGNORECASE)
        ref_tables += re.findall(r'JOIN\s+([a-zA-Z0-9_]+)', ref_query, re.IGNORECASE)
        
        ref_columns = re.findall(r'SELECT\s+(.*?)\s+FROM', ref_query, re.IGNORECASE)
        if ref_columns:
            ref_columns = re.findall(r'([a-zA-Z0-9_]+)(?:\.[a-zA-Z0-9_]+)?', ref_columns[0])
        
        # Remove common SQL keywords from columns
        keywords = ['as', 'distinct', 'count', 'sum', 'avg', 'min', 'max']
        ref_columns = [col for col in ref_columns if col.lower() not in keywords]
        
        # Find tables and columns in prediction
        tables_found = 0
        for table in ref_tables:
            if re.search(rf'\b{table}\b', pred_query, re.IGNORECASE):
                tables_found += 1
        
        columns_found = 0
        for col in ref_columns:
            if re.search(rf'\b{col}\b', pred_query, re.IGNORECASE):
                columns_found += 1
        
        # Calculate scores
        table_score = tables_found / len(ref_tables) if ref_tables else 1.0
        column_score = columns_found / len(ref_columns) if ref_columns else 1.0
        
        # Weighted score (tables are more important)
        return 0.6 * table_score + 0.4 * column_score
    
    except Exception as e:
        print(f"Error in entity extraction: {e}")
        return 0

num_eval_samples = min(100, len(test_data)) 
predictions = []
cleaned_predictions = []
references = []
prompts = []
contexts = []

# Process test samples
for i in tqdm(range(num_eval_samples)):
    try:
        sample = test_data[i]
        
        sql_prompt = sample['sql_prompt']
        sql_context = sample['sql_context']
        reference_sql = sample['sql']
        
        predicted_sql = generate_sql(sql_prompt, sql_context)
        
        cleaned_sql = clean_sql_output(predicted_sql)
        
        if not cleaned_sql:
            print(f"Warning: Cleaning removed all content from sample {i}")
            continue
        
        predictions.append(predicted_sql)
        cleaned_predictions.append(cleaned_sql)
        references.append(reference_sql)
        prompts.append(sql_prompt)
        contexts.append(sql_context)
        
    except Exception as e:
        print(f"Error processing sample {i}: {e}")
        print(f"Sample content: {sample}")
        continue

print(f"Successfully processed {len(predictions)} out of {num_eval_samples} samples")

# Calculate standard metrics using cleaned predictions
if cleaned_predictions:
    results = {
        "bleu": bleu.compute(predictions=cleaned_predictions, references=[[r] for r in references]),
        "rouge": rouge.compute(predictions=cleaned_predictions, references=references),
        "chrf": chrf.compute(predictions=cleaned_predictions, references=references),
    }
    
    # Calculate exact match (case insensitive)
    exact_match_results = exact_match.compute(
        predictions=[q.lower() for q in cleaned_predictions], 
        references=[r.lower() for r in references]
    )
    
    # Calculate normalized exact match with custom preprocessing
    norm_predictions = [normalize_sql_for_comparison(q) for q in cleaned_predictions]
    norm_references = [normalize_sql_for_comparison(r) for r in references]
    
    normalized_exact_match = exact_match.compute(
        predictions=norm_predictions,
        references=norm_references
    )
    
    # Calculate component-based scores
    component_scores = []
    entity_scores = []
    
    for pred, ref in zip(cleaned_predictions, references):
        component_scores.append(calculate_sql_component_match(pred, ref))
        entity_scores.append(contains_key_entities(pred, ref))
    
    # Track per-query metrics
    per_query_metrics = []
    for i, (pred, ref) in enumerate(zip(cleaned_predictions, references)):
        metrics = {
            "query_id": i,
            "exact_match": norm_predictions[i] == norm_references[i],
            "component_match": component_scores[i],
            "entity_match": entity_scores[i],
        }
        per_query_metrics.append(metrics)
    
    # Create a DataFrame to show sample-level predictions and references
    results_df = pd.DataFrame({
        "prompt": prompts,
        "context": [ctx[:100] + "..." if len(ctx) > 100 else ctx for ctx in contexts],  # Truncate context for readability
        "reference": references,
        "original_prediction": predictions,
        "cleaned_prediction": cleaned_predictions,
        "exact_match": [norm_predictions[i] == norm_references[i] for i in range(len(norm_predictions))],
        "component_score": component_scores,
        "entity_score": entity_scores
    })
    
    results_df["cleaning_changed"] = results_df["original_prediction"] != results_df["cleaned_prediction"]
    
    results_df.to_csv("evaluation_results.csv", index=False)
    
    if any(results_df["cleaning_changed"]):
        changed_samples = results_df[results_df["cleaning_changed"]]
        
        cleaning_summary = pd.DataFrame({
            "sample_idx": changed_samples.index,
            "prompt": changed_samples["prompt"],
            "before_cleaning": changed_samples["original_prediction"],
            "after_cleaning": changed_samples["cleaned_prediction"],
        })
        
        cleaning_summary.to_csv("cleaning_changes.csv", index=False)
        print(f"\n{len(changed_samples)} samples were modified during cleaning. See 'cleaning_changes.csv'")

    # Print overall metrics
    print("\nEvaluation Metrics (with cleaned predictions):")
    print(f"BLEU Score: {results['bleu']['bleu']:.4f}")
    print(f"ROUGE-L F1: {results['rouge']['rougeL']:.4f}")
    print(f"CHRF Score: {results['chrf']['score']:.4f}")
    
    # Print SQL-specific metrics
    print("\nSQL-Specific Metrics:")
    print(f"Exact Match (case insensitive): {exact_match_results['exact_match']:.4f}")
    print(f"Normalized Exact Match: {normalized_exact_match['exact_match']:.4f}")
    print(f"Average Component Match: {np.mean(component_scores):.4f}")
    print(f"Average Entity Match: {np.mean(entity_scores):.4f}")
    
    # Calculate success thresholds
    high_quality_count = sum(1 for score in component_scores if score >= 0.8)
    medium_quality_count = sum(1 for score in component_scores if 0.5 <= score < 0.8)
    low_quality_count = sum(1 for score in component_scores if score < 0.5)
    
    print(f"\nQuery Quality Distribution:")
    print(f"High Quality (≥80% component match): {high_quality_count} ({high_quality_count/len(component_scores)*100:.1f}%)")
    print(f"Medium Quality (50-79% component match): {medium_quality_count} ({medium_quality_count/len(component_scores)*100:.1f}%)")
    print(f"Low Quality (<50% component match): {low_quality_count} ({low_quality_count/len(component_scores)*100:.1f}%)")

    # Generate Markdown for the model card
    model_card_metrics = f"""
## Evaluation Metrics

The model was evaluated on {len(cleaned_predictions)} samples from the [gretelai/synthetic_text_to_sql](https://huggingface.co/datasets/gretelai/synthetic_text_to_sql) dataset.

### Standard Text Generation Metrics
| Metric | Score |
|--------|-------|
| BLEU | {results['bleu']['bleu']:.4f} |
| ROUGE-L | {results['rouge']['rougeL']:.4f} |
| CHRF | {results['chrf']['score']:.4f} |

### SQL-Specific Metrics
| Metric | Score |
|--------|-------|
| Exact Match (case insensitive) | {exact_match_results['exact_match']:.4f} |
| Normalized Exact Match | {normalized_exact_match['exact_match']:.4f} |
| Component Match | {np.mean(component_scores):.4f} |
| Entity Match | {np.mean(entity_scores):.4f} |

### Query Quality Distribution
- **High Quality** (≥80% component match): {high_quality_count} ({high_quality_count/len(component_scores)*100:.1f}%)
- **Medium Quality** (50-79% component match): {medium_quality_count} ({medium_quality_count/len(component_scores)*100:.1f}%)
- **Low Quality** (<50% component match): {low_quality_count} ({low_quality_count/len(component_scores)*100:.1f}%)

### Evaluation Details

- **Dataset**: gretelai/synthetic_text_to_sql (test split)
- **Samples**: {len(cleaned_predictions)} examples
- **Input**: SQL prompts and table context
- **Output**: Generated SQL queries (cleaned of comments and redundant queries)
- **Evaluation Date**: {pd.Timestamp.now().strftime('%Y-%m-%d')}

### Metric Explanations

- **Normalized Exact Match**: Percentage of queries that match the reference after normalization (case, whitespace, keyword formatting)
- **Component Match**: Average percentage of SQL components (SELECT, FROM, WHERE, etc.) that match between prediction and reference
- **Entity Match**: Average percentage of table and column entities correctly identified in predictions

### Post-processing
The SQL queries generated by the model were post-processed to:
1. Remove SQL comments
2. Extract the most relevant query when multiple similar queries were generated
3. Normalize formatting for fair comparison with references
"""

    # Save model card metrics to a file
    with open("model_card_metrics_few.md", "w") as f:
        f.write(model_card_metrics)

    print("\nModel card metrics saved to 'model_card_metrics.md'")
else:
    print("No successful predictions were made. Please check the dataset and model.")