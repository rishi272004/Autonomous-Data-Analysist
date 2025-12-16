# tests/test_query_engine.py
def test_parser_import():
    from backend.query_engine.parser import nl_to_sql
    assert callable(nl_to_sql)
