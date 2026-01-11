from bokeh.plotting import figure, show
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource
from bokeh.transform import factor_cmap
from bokeh.palettes import Category10, Category20
import pandas as pd

"""
Bokeh visualizations for training/ideal functions and mapped test points.
"""



class BokehViz:
    """
    Create Bokeh plots for training vs ideal curves and mapped test points.
    """
    def __init__(self, train_dframe, ideal_dframe, best_matches, mapping_dframe = None):
        self.train_dframe = train_dframe
        self.ideal_dframe = ideal_dframe
        self.best_matches = best_matches
        self.mapping_dframe = mapping_dframe

    def training_vs_ideal(self):
        """
        Plot each training function against its selected ideal function.
        """
        all_plots = []

        #loop each matched pair
        for train_func, (ideal_func, error) in self.best_matches.items():

        #create base plot
            p = figure(
                width=400,
                height=300,
                title=f"{train_func} vs {ideal_func}",
                x_axis_label="X",
                y_axis_label="Y",
            )
            
            #training function line draw
            p.line(
                self.train_dframe["x"],
                self.train_dframe[train_func],
                legend_label=f"Train: {train_func}",
                line_width = 2
            )

            #ideal function line draw
            p.line(
                self.ideal_dframe["x"],
                self.ideal_dframe[ideal_func],
                line_dash="dashed",
                legend_label=f"Ideal: {ideal_func}",
                line_color = "orange"
            )

            
            p.legend.location = "top_left"
            all_plots.append(p)

        show(gridplot(all_plots, ncols=2))



    def mapped_test_points(self):
        """
        Plot mapped test points colored by the assigned ideal function.
        """
        source = ColumnDataSource(self.mapping_dframe)
        uniq_func = list(self.mapping_dframe["ideal_function"].unique())

        #choose palette
        if len(uniq_func) <= 10:
            palette = Category10[max(3, len(uniq_func))]
        else:
            palette = Category20[20]

        # create plot base
        p = figure(
            width=700,
            height=450,
            title="Mapped test points by ideal function",
            x_axis_label="X",
            y_axis_label="y",
        )
        
        #scatter with colors
        p.scatter(
            x="x",
            y="y",
            size=10,
            source=source,
            color=factor_cmap("ideal_function", palette=palette, factors=uniq_func),
            legend_field="ideal_function",
        )

        p.legend.location = "top_left"
        show(p)
