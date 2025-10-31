import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset
df = pd.read_csv('malicious_phish.csv')

# Binary encoding dangerous URL types → 1, safe → 0
df["label"] = df['type'].apply(lambda x: 1 if x in ['phishing', 'malware', 'defacement'] else 0)

# Feature flags
df['susp_keyword'] = df['url'].str.contains('login|verify|secure|update', case=False, na=False)
df['unusual_tld'] = df['url'].str.contains(r"\.(?:xyz|info|top|ru|cc|pw|cn|tk)", case=False, na=False)  # Non-capturing group
df['contains_symb'] = df['url'].str.contains(r"[@=\-]", na=False)
df['double_slash'] = df['url'].str.replace(r"[a-z]+://", "", regex=True).str.contains("//", na=False)
df['long_url'] = df["url"].str.len() > 75

# System A: Simple rule-based
df["pred_A"] = (df[['susp_keyword', "unusual_tld", 'contains_symb', 'double_slash', 'long_url']].sum(axis=1) > 2).astype(int)

# System B: Weighted features 
df['pred_B'] = (
    (df['susp_keyword'].astype(int) * 2 +
     df['unusual_tld'].astype(int) * 3 +
     df['contains_symb'].astype(int) * 5 +
     df['double_slash'].astype(int) * 2 +
     df['long_url'].astype(int) * 3) >= 8
).astype(int)

# Metrics function
def metrics(name, y_true, y_pred):
    print(f"\n{name}")
    print(f"Accuracy Score: {accuracy_score(y_true, y_pred):.3f}")
    print(f"Precision Score: {precision_score(y_true, y_pred, zero_division=0):.3f}")
    print(f"Recall: {recall_score(y_true, y_pred, zero_division=0):.3f}")
    print(f"F1 Score: {f1_score(y_true, y_pred, zero_division=0):.3f}")

# Evaluate
metrics("AI A", df["label"], df["pred_A"])
metrics("AI B", df["label"], df['pred_B'])