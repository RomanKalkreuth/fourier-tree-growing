import plotly.express as px
import plotly.graph_objects as go

def line_plot(x, y, title=""):
    """

    """
    fig = px.line(x, y, title)
    fig.show()


def surface_plot(x, y, z, title=""):
    """

    """
    fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])
    fig.show()

def scatter2d(x, y):
    """

    """
    fig = px.scatter(x, y)
    fig.show()
