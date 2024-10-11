import numpy as np
from bokeh.models import Select, HoverTool, Row
from bokeh.plotting import figure

from .bokeh_function_visualiser import BokehFunctionVisualiser


class BokehFunctionVisualiser1D(BokehFunctionVisualiser):
    """
    1D implementation of BokehFunctionVisualiser
    """
    def update_input_axes(self) -> None:
        """
        Updates the label and range of the plot's x-axis
        """
        if not self.freeze_input_axes_switch.active:
            self.figure.x_range.update(start=self.xbins[0], end=self.xbins[-1])

        self.figure.xaxis.axis_label = self.input_scalar1.latex_label
        self.figure.yaxis.axis_label = self.output_scalar.latex_label

    def update_output_axes(self) -> None:
        """
        Updates the label and range of the plot's y-axis
        """
        y_min, y_max = self.output_scalar_range(self.last_output)
        self.figure.y_range.update(start=y_min, end=y_max)

    def update_output(self) -> None:
        """
        Re-computes the output of the function and updates the data in the line
        """
        eval_point = np.asarray(list(slider.value for slider in self.sliders))
        input = np.tile(eval_point, (self.xbins.shape[0], 1))
        input[:, self.input_scalar_ind1] = self.xbins
        self.last_output = self.output_scalar(self.function(input))
        self.line.data_source.data = {
            'x': self.xbins,
            'y': self.last_output,
        }
        super().update_output()

    def setup_figure(self) -> None:
        """
        Configures a single figure with a line glyph for rendering a 1D slice of the function
        """
        hover = HoverTool(
            tooltips=[
                ("x", "$x"),
                ("y", "$y"),
            ]
        )
        self.figure = figure(
            x_range=(0, 1),
            y_range=(0, 1),
            sizing_mode='scale_both',
            width=100,
            height=100,
            x_axis_label='x',
            y_axis_label='y'
        )
        self.figure.add_tools(hover)
        self.line = self.figure.line()

    def setup(self) -> None:
        """
        Configures a root 'Row' containing the figure and settings column
        """
        super().setup()
        self.root = Row(self.figure, self.settings_column, sizing_mode='stretch_both')
