# backend/export/report_generator.py
# For a starter, we will create a simple markdown report
def generate_markdown_report(title: str, summary: dict, sample_rows=None) -> str:
    md = [f"# {title}\n"]
    md.append("## Summary\n")
    for k, v in summary.items():
        md.append(f"- **{k}**: {v}\n")
    if sample_rows is not None:
        md.append("\n## Sample rows\n")
        md.append(sample_rows.to_markdown())
    return "\n".join(md)
