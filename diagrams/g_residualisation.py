import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# =========================
# Helper functions
# =========================
def add_soft_arrow(
    fig, start, end, color, row, col,
    width=5, head_len=0.28, head_width=0.16,
    tip_dot=False, tip_dot_size=14
):
    """
    Rounded/cute arrow made from:
    - a thick shaft
    - a soft V-shaped arrowhead
    - optional round dot at the tip
    """
    x0, y0 = start
    x1, y1 = end

    dx, dy = x1 - x0, y1 - y0
    L = np.hypot(dx, dy)
    if L == 0:
        return

    ux, uy = dx / L, dy / L
    px, py = -uy, ux  # perpendicular

    # base point of arrowhead
    bx, by = x1 - head_len * ux, y1 - head_len * uy

    # two soft wings
    lx, ly = bx + head_width * px, by + head_width * py
    rx, ry = bx - head_width * px, by - head_width * py

    # shaft
    fig.add_trace(
        go.Scatter(
            x=[x0, x1],
            y=[y0, y1],
            mode="lines",
            line=dict(color=color, width=width),
            hoverinfo="skip",
            showlegend=False
        ),
        row=row, col=col
    )

    # arrow head
    fig.add_trace(
        go.Scatter(
            x=[lx, x1, rx],
            y=[ly, y1, ry],
            mode="lines",
            line=dict(color=color, width=width),
            hoverinfo="skip",
            showlegend=False
        ),
        row=row, col=col
    )

    # optional rounded tip
    if tip_dot:
        fig.add_trace(
            go.Scatter(
                x=[x1],
                y=[y1],
                mode="markers",
                marker=dict(size=tip_dot_size, color=color),
                hoverinfo="skip",
                showlegend=False
            ),
            row=row, col=col
        )


def add_label(fig, x, y, text, color, row, col, size=22):
    fig.add_trace(
        go.Scatter(
            x=[x],
            y=[y],
            mode="text",
            text=[text],
            textfont=dict(
                family="Arial, Helvetica, sans-serif",
                size=size,
                color=color
            ),
            hoverinfo="skip",
            showlegend=False
        ),
        row=row, col=col
    )


def add_axes(fig, row, col, xlim=(-0.6, 6.2), ylim=(-0.6, 5.2)):
    # x-axis
    add_soft_arrow(
        fig,
        start=(xlim[0], 0),
        end=(xlim[1], 0),
        color="#2f2f2f",
        row=row,
        col=col,
        width=4,
        head_len=0.25,
        head_width=0.12,
        tip_dot=False
    )

    # y-axis
    add_soft_arrow(
        fig,
        start=(0, ylim[0]),
        end=(0, ylim[1]),
        color="#2f2f2f",
        row=row,
        col=col,
        width=4,
        head_len=0.25,
        head_width=0.12,
        tip_dot=False
    )

    # origin dot
    fig.add_trace(
        go.Scatter(
            x=[0],
            y=[0],
            mode="markers",
            marker=dict(size=6, color="#2f2f2f"),
            hoverinfo="skip",
            showlegend=False
        ),
        row=row, col=col
    )


# =========================
# Data
# =========================
origin = np.array([0.0, 0.0])

joy = np.array([2.8, 4.0])
disgust = np.array([4.8, 3.8])
sadness = np.array([4.6, 1.3])

g = np.array([3.1, 2.2])

# residual vectors
r_joy = joy - g
r_disgust = disgust - g
r_sadness = sadness - g

colors = {
    "joy": "#f39c12",
    "disgust": "#76c442",
    "sadness": "#1f77ff",
    "g": "#b3b3b3",
    "black": "#2f2f2f"
}

# =========================
# Figure
# =========================
fig = make_subplots(
    rows=1,
    cols=2,
    horizontal_spacing=0.14
)

# ----- Left panel -----
add_axes(fig, 1, 1)

add_soft_arrow(fig, origin, joy, colors["joy"], 1, 1, width=5, tip_dot=False)
add_soft_arrow(fig, origin, disgust, colors["disgust"], 1, 1, width=5, tip_dot=False)
add_soft_arrow(fig, origin, sadness, colors["sadness"], 1, 1, width=5, tip_dot=False)
add_soft_arrow(fig, origin, np.array([4.5, 2.4]), colors["black"], 1, 1, width=4)

add_label(fig, joy[0]-0.15, joy[1]+0.25, "Joy", colors["joy"], 1, 1)
add_label(fig, disgust[0]+0.15, disgust[1]+0.25, "Disgust", colors["disgust"], 1, 1)
add_label(fig, sadness[0]+0.25, sadness[1]+0.25, "Sadness", colors["sadness"], 1, 1)

# ----- Right panel -----
add_axes(fig, 1, 2)

# common direction g
add_soft_arrow(fig, origin, g, colors["g"], 1, 2, width=5, tip_dot=False)
add_label(fig, g[0]-0.55, g[1]-0.55, "g", colors["g"], 1, 2)

# reference black direction
add_soft_arrow(fig, origin, np.array([4.4, 2.4]), colors["black"], 1, 2, width=4)

# residuals starting from g
add_soft_arrow(fig, g, g + r_joy, colors["joy"], 1, 2, width=5, tip_dot=False)
add_soft_arrow(fig, g, g + r_disgust, colors["disgust"], 1, 2, width=5, tip_dot=False)
add_soft_arrow(fig, g, g + r_sadness, colors["sadness"], 1, 2, width=5, tip_dot=False)

add_label(fig, (g + r_joy)[0]-0.35, (g + r_joy)[1]+0.25, "r<sub>joy</sub>", colors["joy"], 1, 2)
add_label(fig, (g + r_disgust)[0]+0.35, (g + r_disgust)[1]+0.2, "r<sub>disgust</sub>", colors["disgust"], 1, 2)
add_label(fig, (g + r_sadness)[0]+0.45, (g + r_sadness)[1]-0.05, "r<sub>sadness</sub>", colors["sadness"], 1, 2)

# middle arrow between panels
fig.add_annotation(
    x=0.5, y=0.5,
    xref="paper", yref="paper",
    text="➜",
    showarrow=False,
    font=dict(
        family="Arial, Helvetica, sans-serif",
        size=80,
        color="#4a4a4a"
    )
)

# =========================
# Layout
# =========================
for c in [1, 2]:
    fig.update_xaxes(
        row=1, col=c,
        range=[-0.6, 6.2],
        showgrid=True,
        gridcolor="rgba(0,0,0,0.18)",
        gridwidth=1,
        dtick=0.5,
        zeroline=False,
        showticklabels=False,
        showline=False
    )
    fig.update_yaxes(
        row=1, col=c,
        range=[-0.6, 5.2],
        showgrid=True,
        gridcolor="rgba(0,0,0,0.18)",
        gridwidth=1,
        dtick=0.5,
        zeroline=False,
        showticklabels=False,
        showline=False,
        scaleanchor=f"x{c}",
        scaleratio=1
    )

fig.update_layout(
    width=1500,
    height=550,
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(l=20, r=20, t=20, b=20),
    font=dict(
        family="Arial, Helvetica, sans-serif",
        size=18,
        color="#2f2f2f"
    )
)

fig.show()