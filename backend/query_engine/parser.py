# backend/query_engine/parser.py
from backend.llm.ollama_client import query_gemini
from backend.utils.helpers import sanitize_column_name
from backend.utils.logger import get_logger
import re
from typing import Optional
import difflib
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = get_logger(__name__)

def natural_to_sql(natural_query: str, table_name: str, columns: list, context: Optional[str] = None) -> Optional[str]:
    col_map = {col: sanitize_column_name(col) for col in columns}
    sanitized_columns = list(set(col_map.values()))
    
    # Perfect prompt for LLM to handle all scenarios
    perfect_prompt = f"""You are an expert SQL analyst and data scientist. Convert this natural language request into a perfect MySQL 8.x SELECT query.

BUSINESS REQUEST: {natural_query}

DATABASE SCHEMA:
- Table: `{table_name}`
- Available Columns: {', '.join(sanitized_columns)}

CONTEXT: {context or 'No additional context provided'}

ANALYSIS REQUIREMENTS:
1. **Understand the Business Need**: Analyze what the user is trying to achieve
2. **Identify Required Data**: Determine which columns and calculations are needed
3. **Choose Appropriate SQL Patterns**: Use the right SQL constructs (JOINs, subqueries, window functions, etc.)
4. **Handle Complex Scenarios**: 
   - Simulations: Use mathematical calculations (e.g., sales * 1.15 for 15% increase)
   - Comparisons: Use UNION ALL or conditional aggregation
   - Rankings: Use ROW_NUMBER() or RANK() window functions
   - Time analysis: Use DATE_FORMAT() for grouping
   - Geographic analysis: Use appropriate filtering and grouping
5. **Ensure Accuracy**: Validate column names exist and calculations are correct
6. **Optimize Performance**: Write efficient queries with proper indexing considerations

SQL REQUIREMENTS:
- Use only the provided table and columns
- Handle all edge cases and complex business logic
- Use proper GROUP BY when using aggregate functions
- Apply appropriate WHERE clauses for filtering
- Use ORDER BY for rankings and recommendations
- Use LIMIT for top-N results when appropriate
- Ensure MySQL 8.x compatibility
- Output ONLY the SQL query ending with semicolon

EXAMPLES OF COMPLEX SCENARIOS YOU SHOULD HANDLE:
- "Simulate 15% sales increase" → Calculate new values using mathematical operations
- "Recommend top 3 cities" → Use ranking and filtering
- "Compare categories" → Use comparative analysis with ratios
- "What-if scenarios" → Use conditional calculations
- "Impact analysis" → Use subqueries for baseline comparisons
- "Trend analysis" → Use time-based grouping

CRITICAL OUTPUT FORMAT:
- Start your response with SELECT (not "SELECT query" or any other text)
- End with semicolon
- Do not include any explanatory text before or after the SQL
- Do not include comments or descriptions

Generate the perfect SQL query that addresses the business need completely and accurately."""
    
    try:
        # Generate multiple candidate SQLs concurrently and pick the first valid one
        def gen_primary():
            return query_gemini(perfect_prompt)

        # A simplified variant that nudges the model to avoid complex constructs
        simple_prompt = perfect_prompt + "\n\nConstraints: Avoid CTEs, avoid nested subqueries in SELECT, prefer GROUP BY + ORDER BY + LIMIT."
        def gen_simple():
            return query_gemini(simple_prompt)

        # A repair prompt that asks to fix only syntax/compatibility to MySQL 8.x
        def gen_repair(seed_sql: str) -> str:
            repair_prompt = f"""Fix this to valid MySQL 8.x SELECT without changing intent.

Rules: no CTEs; no parentheses around UNION parts; parenthesize scalar subqueries in SELECT; end with semicolon; single table `{table_name}` and columns only: {', '.join(sanitized_columns)}.

SQL:
{seed_sql}
"""
            return query_gemini(repair_prompt)

        candidates_raw = []
        with ThreadPoolExecutor(max_workers=3) as ex:
            futs = [ex.submit(gen_primary), ex.submit(gen_simple)]
            for f in as_completed(futs):
                try:
                    out = f.result()
                    if out:
                        candidates_raw.append(out)
                except Exception:
                    continue
        # Attempt repairs on raw candidates in parallel if needed
        with ThreadPoolExecutor(max_workers=3) as ex:
            repair_futs = [ex.submit(gen_repair, c) for c in candidates_raw]
            for f in as_completed(repair_futs):
                try:
                    out = f.result()
                    if out:
                        candidates_raw.append(out)
                except Exception:
                    continue

        # Deduplicate while preserving order
        seen = set()
        ordered_candidates = []
        for c in candidates_raw:
            k = c.strip()
            if k not in seen:
                seen.add(k)
                ordered_candidates.append(k)

        def normalize_sql(sql_text: str) -> Optional[str]:
            if not sql_text:
                return None
            s = re.sub(r"```(?:sql)?", "", sql_text, flags=re.IGNORECASE).strip()
            s = re.sub(r"^.*?(?=SELECT)", "", s, flags=re.IGNORECASE | re.DOTALL).strip()
            m = re.search(r"(SELECT[\s\S]*?;)", s, flags=re.IGNORECASE)
            s = m.group(1).strip() if m else s
            if not s.upper().startswith('SELECT'):
                s = re.sub(r"^.*?(SELECT)", r"\1", s, flags=re.IGNORECASE | re.DOTALL)
            if not s.endswith(';'):
                s += ';'
            s = validate_and_fix_sql(s)
            for original, sanitized in col_map.items():
                s = s.replace(original, sanitized)
            s = normalize_sql_to_schema(s, table_name, sanitized_columns)
            s = rewrite_sql_for_mysql_compat(s)
            if not re.search(r"\bFROM\b", s, flags=re.IGNORECASE):
                s = re.sub(r";+$", "", s).strip() + f" FROM {table_name};"
            # Final sanity: looks like single SELECT and balanced parentheses
            if re.search(r"\)\s*SELECT\s", s, flags=re.IGNORECASE):
                return None
            bal = 0
            for ch in s:
                if ch == '(': bal += 1
                elif ch == ')': bal -= 1
            if bal < 0:
                return None
            return s

        for raw in ordered_candidates:
            normalized = normalize_sql(raw)
            if normalized:
                logger.info("SQL generated successfully")
                return normalized

        # If none normalize cleanly, last attempt: use simple prompt result normalized best-effort
        if ordered_candidates:
            fallback_norm = normalize_sql(ordered_candidates[-1])
            if fallback_norm:
                logger.info("SQL generated successfully (fallback normalized)")
                return fallback_norm

        # As a final safety, try heuristic (will still respect schema)
        return build_heuristic_sql(natural_query, table_name, sanitized_columns)
    except Exception as e:
        logger.error(f"SQL generation failed: {e}")
        # Fallback on error
        return build_heuristic_sql(natural_query, table_name, sanitized_columns)

def rewrite_sql_for_mysql_compat(sql: str) -> str:
    """
    Rewrites patterns MySQL may not support, e.g., LIMIT inside IN subqueries, by converting to JOIN on a derived table.
    Example transform:
      WHERE a IN (SELECT a FROM t GROUP BY a ORDER BY SUM(x) DESC LIMIT 5)
      => JOIN (SELECT a FROM t GROUP BY a ORDER BY SUM(x) DESC LIMIT 5) AS __top_ids ON main.a = __top_ids.a
    """
    try:
        # Detect main SELECT ... FROM main_table ... WHERE main_col IN ( <inner_select> ) ;
        pattern = re.compile(r"SELECT\s+(?P<select>.+?)\s+FROM\s+(?P<main>\w+)\s+(?P<rest>.*?)(WHERE\s+(?P<lhs>\w+(?:\.\w+)?))\s+IN\s*\((?P<inner>SELECT[\s\S]+?LIMIT\s+\d+\s*)\)\s*;?\s*$",
                              re.IGNORECASE)
        m = pattern.search(sql)
        if not m:
            return sql
        select_list = m.group('select')
        main_table = m.group('main')
        rest_before_where = m.group('rest')
        lhs_col = m.group('lhs')
        inner_select = m.group('inner').strip()
        # Extract the projected column name from inner select: SELECT <col> FROM ...
        inner_proj = re.search(r"SELECT\s+(?P<icol>\w+(?:\.\w+)?)\s+FROM", inner_select, flags=re.IGNORECASE)
        inner_col = inner_proj.group('icol') if inner_proj else lhs_col
        derived_alias = "__top_ids"
        # Build JOIN form
        new_sql = f"SELECT {select_list} FROM {main_table} {rest_before_where}JOIN ( {inner_select} ) AS {derived_alias} ON {lhs_col} = {derived_alias}.{inner_col.split('.')[-1]}"
        # Preserve trailing semicolon
        if not new_sql.strip().endswith(';'):
            new_sql += ';'
        return new_sql
    except Exception:
        return sql


def validate_and_fix_sql(sql: str) -> str:
    """
    Basic SQL validation and cleanup.
    """
    try:
        # Basic cleanup
        sql = re.sub(r'\s+', ' ', sql)  # Normalize whitespace
        sql = sql.strip()

        # Quick rejection of unsupported CTEs (WITH ...)
        if re.search(r'^\s*WITH\s+', sql, flags=re.IGNORECASE):
            raise ValueError('CTE not supported')

        # Fix extra parentheses around UNION parts
        sql = re.sub(r"\)\s+UNION\s+ALL\s+\(", " UNION ALL ", sql, flags=re.IGNORECASE)

        # Fix missing opening parenthesis before aliased scalar subqueries/aggregates inside SELECT list
        # General form inside SELECT list:  <expr FROM ...>) AS alias  → (<expr FROM ...>) AS alias
        pat_scalar_alias = re.compile(r"(?:(?:^|,)[\s]*)(?!\()([\s\S]*?\bFROM\s+\w+[\s\S]*?)\)\s+AS\s+(\w+)", re.IGNORECASE)
        def _wrap_scalar_alias(m):
            inner = m.group(1).strip()
            alias = m.group(2)
            return f" ({inner}) AS {alias}"
        # Apply iteratively to catch multiple items
        prev = None
        while prev != sql:
            prev = sql
            sql = pat_scalar_alias.sub(_wrap_scalar_alias, sql)

        # If multiple top-level SELECTs glued (e.g., ") SELECT" after a subquery), fallback to safe pattern
        if re.search(r"\)\s*SELECT\s", sql, flags=re.IGNORECASE):
            raise ValueError('Multiple top-level SELECTs detected')

        # Ensure proper semicolon
        if not sql.endswith(';'):
            sql += ';'

        # Simple parentheses balance check; if broken, raise to trigger fallback
        balance = 0
        for ch in sql:
            if ch == '(': balance += 1
            elif ch == ')': balance -= 1
        if balance < 0:
            raise ValueError('Unbalanced parentheses')
        return sql
    except Exception:
        return sql

def normalize_sql_to_schema(sql: str, table_name: str, valid_columns: list) -> str:
    """
    Normalize LLM SQL to the known single-table schema to reduce hallucinations:
    - Ensure FROM uses the active table
    - Map unknown identifiers to the closest valid column using difflib
    - Prevent references to unknown tables
    """
    try:
        s = sql
        # Force FROM table_name for single-table context
        if not re.search(r"\bFROM\s+\w+", s, flags=re.IGNORECASE):
            s = re.sub(r";+$", "", s).strip() + f" FROM {table_name};"
        else:
            s = re.sub(r"(FROM\s+)(\w+)", fr"\1{table_name}", s, flags=re.IGNORECASE)

        # Attempt to correct bare identifiers that are not functions/keywords
        tokens = re.split(r"([^\w\.]+)", s)
        corrected = []
        for t in tokens:
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", t):
                # Skip SQL keywords and table name
                if t.upper() in {"SELECT","FROM","WHERE","GROUP","BY","ORDER","LIMIT","AS","AND","OR","DESC","ASC","ON","JOIN","LEFT","RIGHT","INNER","OUTER","HAVING","COUNT","SUM","AVG","MIN","MAX","DISTINCT","CASE","WHEN","THEN","ELSE","END","DATE_FORMAT"}:
                    corrected.append(t)
                    continue
                if t == table_name:
                    corrected.append(t)
                    continue
                # If identifier not in valid columns, map to closest
                if t not in valid_columns:
                    match = difflib.get_close_matches(t, valid_columns, n=1, cutoff=0.75)
                    corrected.append(match[0] if match else t)
                else:
                    corrected.append(t)
            else:
                corrected.append(t)
        s = ''.join(corrected)
        return s
    except Exception:
        return sql

def build_heuristic_sql(natural_query: str, table_name: str, columns: list) -> Optional[str]:
    """
    Heuristic SQL builder as a fallback when LLM output is invalid.
    Covers common intents: totals/averages, top/bottom rankings, and category breakdowns.
    """
    try:
        q = (natural_query or '').lower()
        cols_lower = {c.lower(): c for c in columns}

        def find_col(candidates):
            for cand in candidates:
                for col in columns:
                    if cand in col.lower():
                        return col
            return None

        metric_sales = find_col(['sales', 'revenue', 'amount'])
        metric_profit = find_col(['profit', 'margin'])
        metric_qty = find_col(['quantity', 'qty'])
        dim_product = find_col(['product_name', 'product', 'item', 'sku'])
        dim_customer = find_col(['customer_name', 'customer'])
        dim_city = find_col(['city'])
        dim_state = find_col(['state'])
        dim_region = find_col(['region'])
        dim_category = find_col(['category'])
        dim_sub_category = find_col(['sub_category', 'sub category'])
        dim_any = next((d for d in [dim_product, dim_customer, dim_city, dim_state, dim_region, dim_sub_category, dim_category] if d), None)

        # Parse desired limit
        import re
        m = re.search(r'(top|bottom)\s+(\d+)', q)
        limit_n = int(m.group(2)) if m else 10
        want_top = bool(m and m.group(1) == 'top')
        want_bottom = bool(m and m.group(1) == 'bottom')

        # Totals / averages intent
        if any(k in q for k in ['total', 'overall', 'sum']) and any(k in q for k in ['average', 'avg']) and (metric_sales or metric_profit):
            parts = []
            if metric_sales:
                parts.append(f"SUM({metric_sales}) AS total_sales")
            if metric_profit:
                parts.append(f"AVG({metric_profit}) AS average_{metric_profit}")
            if parts:
                return f"SELECT {', '.join(parts)} FROM {table_name};"

        if any(k in q for k in ['total', 'sum']) and metric_sales and not dim_any:
            return f"SELECT SUM({metric_sales}) AS total_sales FROM {table_name};"

        if any(k in q for k in ['average', 'avg']) and metric_profit and not dim_any:
            return f"SELECT AVG({metric_profit}) AS average_{metric_profit} FROM {table_name};"

        # Top/bottom rankings by a dimension
        metric = metric_sales or metric_profit or metric_qty
        if (want_top or want_bottom) and dim_any and metric:
            order = 'DESC' if want_top else 'ASC'
            return (
                f"SELECT {dim_any} AS dimension, SUM({metric}) AS total_metric "
                f"FROM {table_name} GROUP BY {dim_any} ORDER BY total_metric {order} LIMIT {limit_n};"
            )

        # Generic breakdown when asking by/category
        if any(k in q for k in ['by ', ' per ', 'group', 'breakdown', 'segment']) and dim_any and metric:
            return (
                f"SELECT {dim_any} AS dimension, SUM({metric}) AS total_metric, AVG({metric}) AS average_metric "
                f"FROM {table_name} GROUP BY {dim_any} ORDER BY total_metric DESC LIMIT {limit_n};"
            )

        # Last resort: simple totals for whatever metrics found
        parts = []
        if metric_sales:
            parts.append(f"SUM({metric_sales}) AS total_sales")
        if metric_profit:
            parts.append(f"SUM({metric_profit}) AS total_profit")
        if metric_qty:
            parts.append(f"SUM({metric_qty}) AS total_quantity")
        return f"SELECT {', '.join(parts) if parts else 'COUNT(*) AS total_rows'} FROM {table_name};"
    except Exception:
        return None