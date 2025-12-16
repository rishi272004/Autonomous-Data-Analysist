# backend/llm/prompt_templates.py
# Keep prompt templates centralized
SQL_TRANSLATION_PROMPT = """
You are given a database schema and a user's analytic request.
Translate the request into a valid MySQL 8.x SELECT query using exact table/column names.
Rules:
- Use GROUP BY for non-aggregated selected columns when using aggregates
- Use DATE_FORMAT for time grouping when a date column exists
- Avoid LIMIT inside IN subqueries; prefer JOIN with a derived table
- Respect ONLY_FULL_GROUP_BY; no ambiguous selects
Return ONLY a valid JSON object: {"sql": "SELECT ...;"}
"""

EXPLANATION_BULLETS_PROMPT = """
You are a senior business analyst. Produce 4–6 hyphen bullets with concrete numbers based on the provided data sample and query. No headings, no asterisks, no numbering, no meta commentary. Each bullet under 25 words.
Inputs:
- Query: {{user_query}}
- Rows: {{n_rows}}, Cols: {{n_cols}}
- Sample:
{{sample_rows_text}}
"""
