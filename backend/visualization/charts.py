import plotly.express as px

def visualize(df, chart_type="bar", x=None, y=None):
    if chart_type == "bar":
        fig = px.bar(df, x=x, y=y)
    elif chart_type == "line":
        fig = px.line(df, x=x, y=y)
    elif chart_type == "pie":
        fig = px.pie(df, names=x, values=y)
    else:
        return None
    return fig
