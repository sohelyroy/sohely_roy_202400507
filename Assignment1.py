# Assignment 1: Market Basket Analysis

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Sample real-world style shopping transactions
dataset = [
    ['Milk', 'Bread', 'Butter'],
    ['Bread', 'Butter'],
    ['Milk', 'Bread'],
    ['Milk', 'Eggs'],
    ['Bread', 'Eggs'],
    ['Milk', 'Bread', 'Eggs'],
    ['Bread', 'Butter', 'Jam'],
    ['Milk', 'Bread', 'Butter'],
    ['Milk', 'Eggs', 'Butter'],
    ['Bread', 'Jam']
]

# Convert transactions into one-hot encoded format
te = TransactionEncoder()
te_array = te.fit(dataset).transform(dataset)

df = pd.DataFrame(te_array, columns=te.columns_)

print("Transaction Data:")
print(df)

# Find frequent itemsets
frequent_itemsets = apriori(
    df,
    min_support=0.3,
    use_colnames=True
)

print("\nFrequent Itemsets:")
print(frequent_itemsets)

# Generate association rules
rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=0.5
)

print("\nAssociation Rules:")
print(rules[['antecedents',
             'consequents',
             'support',
             'confidence',
             'lift']])
