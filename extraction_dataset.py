from datasets import load_dataset
import re

print("Loading LMSYS Chat dataset...")
ds = load_dataset("lmsys/lmsys-chat-1m", split="train")
print(ds)

# ---- Step 1. Extract user queries ----
def extract_queries(example):
    queries = []
    for conv in example["conversation"]:
        if isinstance(conv, dict) and conv.get("role") == "user":
            queries.append(conv["content"])
    return {"user_queries": queries}

print("Extracting user queries...")
user_ds = ds.map(extract_queries, remove_columns=ds.column_names)

# ---- Step 2. Flatten lists properly ----
def split_queries(batch):
    queries = []
    for lst in batch["user_queries"]:
        if lst:
            queries.extend(lst)
    return {"user_query": queries}

user_ds = user_ds.map(split_queries, batched=True, remove_columns=["user_queries"])
print("Total user queries:", len(user_ds))

# ---- Step 3. Define keywords ----
keywords = [
    "python", "java", "c++", "c#", "javascript", "sql", "bash",
    "code", "function", "class", "method", "script",
    "debug", "compile", "program", "algorithm"
]

def is_coding_example(example):
    query = example["user_query"]
    if not isinstance(query, str):
        return False
    q = query.lower()
    return any(kw in q for kw in keywords)

print("Filtering coding-related queries...")
coding_ds = user_ds.filter(is_coding_example)
print("Found coding queries:", len(coding_ds))

# ---- Step 4. Print sample ----
print("\n=== 10 code-related prompts (random) ===")
for row in coding_ds.shuffle(seed=42).select(range(10)):
    print("-", row["user_query"])

# ---- Step 5. Save to CSV ----
output_path = "coding_queries.csv"
coding_ds.to_csv(output_path, escapechar="\\")  # 添加 escapechar
print(f"\nSaved filtered queries to {output_path}")