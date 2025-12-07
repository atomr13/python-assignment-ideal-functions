from bokeh.plotting import figure, show
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource
from bokeh.transform import factor_cmap
from bokeh.palettes import Category10

import pandas as pd
from .exceptions import DataMismatchError

class VisualizationManager:
    """
    Handles Bokeh visualizations for:
    - training vs ideal functions
    - mapped test points
    """
    def __init__(
        self,
        train_df: pd.DataFrame,
        ideal_df: pd.DataFrame,
        best_matches: dict,
        mapping_df: pd.DataFrame | None = None,
    ):
        self.train_df = train_df
        self.ideal_df = ideal_df
        self.best_matches = best_matches
        self.mapping_df = mapping_df

    def plot_training_vs_ideal(self):
        """
        Create a 2x2 grid of plots: each training function vs its matched ideal function.
        """
        plots = []

        for t_col, (i_col, _) in self.best_matches.items():
            p = figure(
                width=400,
                height=300,
                title=f"{t_col} vs {i_col}",
                x_axis_label="x",
                y_axis_label="y",
            )

            # training data
            p.line(self.train_df["x"], self.train_df[t_col],
                   legend_label=f"train {t_col}")

            # ideal function (dashed)
            p.line(self.ideal_df["x"], self.ideal_df[i_col],
                   line_dash="dashed",
                   legend_label=f"ideal {i_col}")

            p.legend.location = "top_left"
            plots.append(p)

        # assume exactly 4 training functions (y1..y4)
        grid = gridplot([[plots[0], plots[1]],
                         [plots[2], plots[3]]])

        show(grid)

    def plot_mapped_test_points(self):
        """
        Scatter plot of mapped test points, colored by ideal_function.
        """
        if self.mapping_df is None or self.mapping_df.empty:
            raise DataMismatchError("mapping_df is empty. Run TestMapper.map_test_points() first.")

        source = ColumnDataSource(self.mapping_df)
        factors = list(self.mapping_df["ideal_function"].unique())

        p = figure(
            width=700,
            height=450,
            title="Mapped test points by ideal function",
            x_axis_label="x",
            y_axis_label="y",
        )

        # Use scatter (Bokeh 3+ friendly) instead of circle
        p.scatter(
            x="x",
            y="y",
            size=10,
            source=source,
            color=factor_cmap(
                "ideal_function",
                palette=Category10[max(3, len(factors))],
                factors=factors,
            ),
            legend_field="ideal_function",
        )

        p.legend.location = "top_left"
        show(p)
