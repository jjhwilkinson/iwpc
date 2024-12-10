import numpy as np
from bokeh.models import Select, HoverTool, Row
from bokeh.plotting import figure

from .bokeh_function_visualiser import BokehFunctionVisualiser


class BokehFunctionVisualiser1D(BokehFunctionVisualiser):
    def __init__(
        self,
        *args,
        use_points: bool = False,
        initial_x_axis_scalar_ind: int = 0,
        **kwargs
    ):
        """
        Parameters
        ----------
        args
            Any BokehFunctionVisualiser arguments
        use_points
            Whether the data should be rendered as a line or with points
        kwargs
            Any BokehFunctionVisualiser keyword arguments
        initial_x_axis_scalar_ind
            The index of the x-axis scalar to initially select on start up
        """
        self.use_points = use_points
        self.initial_x_axis_scalar_ind = initial_x_axis_scalar_ind
        super().__init__(*args, **kwargs)

    """
    1D implementation of BokehFunctionVisualiser
    """
    def update_input_axes(self) -> None:
        """
        Updates the label and range of the plot's x-axis
        """
        self.main_figure.x_range.update(
            reset_start=self.xbins[0],
            reset_end=self.xbins[-1],
        )

        if not self.freeze_input_axes_switch.active:
            self.main_figure.x_range.update(
                start=self.xbins[0],
                end=self.xbins[-1]
            )

        self.main_figure.xaxis.axis_label = self.input_scalar1.latex_label
        self.main_figure.yaxis.axis_label = self.output_scalar.latex_label

    def update_output_axes(self) -> None:
        """
        Updates the label and range of the plot's y-axis
        """
        y_min, y_max = self.output_scalar_range(self.last_scalar_output)

        self.main_figure.y_range.update(
            reset_start=y_min,
            reset_end=y_max,
        )

        if not self.freeze_output_axes_switch.active:
            self.main_figure.y_range.update(
                start=y_min,
                end=y_max,
                reset_start=y_min,
                reset_end=y_max,
            )

    def _update_output(self, reuse_previous_output: bool = False) -> None:
        """
        Re-computes the output of the function and updates the data in the line

        Parameters
        ----------
        reuse_previous_output
            Whether to reuse the outputs of the previous function evaluation. Useful if the user has performed some
            action that would not affect the output of the function such as changing the output scalar
        """
        if not reuse_previous_output:
            eval_point = np.asarray(list(slider.value for slider in self.sliders))
            input = np.tile(eval_point, (self.xbins.shape[0], 1))
            input[:, self.input_scalar_ind1] = self.xbins
            self.last_output = self.function(input)
        self.last_scalar_output = self.output_scalar(self.last_output)
        super()._update_output()

    def update_output(self, reuse_previous_output: bool = False) -> None:
        """
        Updates the output of the function and the plots

        Parameters
        ----------
        reuse_previous_output
            Whether to reuse the outputs of the previous function evaluation. Useful if the user has performed some
            action that would not affect the output of the function such as changing the output scalar
        """
        super().update_output(reuse_previous_output)
        self.line.data_source.data = {
            'x': self.xbins,
            'y': self.last_scalar_output,
        }

    def setup_figure(self) -> None:
        """
        Configures a single figure with a line glyph for rendering a 1D slice of the function
        """
        hover = HoverTool(
            tooltips=[
                ("x", "@x"),
                ("y", "@y"),
            ],
            point_policy='snap_to_data',
            line_policy='interp',
        )
        self.main_figure = figure(
            x_range=(0, 1),
            y_range=(0, 1),
            sizing_mode='scale_both',
            width=100,
            height=100,
            x_axis_label='x',
            y_axis_label='y'
        )
        self.main_figure.add_tools(hover)

        if self.use_points:
            self.line = self.main_figure.scatter(line_color="#3288bd", fill_color="white", line_width=2)
        else:
            self.line = self.main_figure.line()


    def setup(self) -> None:
        """
        Configures a root 'Row' containing the figure and settings column
        """
        super().setup()
        self.root = Row(self.main_figure, self.settings_column, sizing_mode='stretch_both')

    def setup_input_scalar_pickers(self) -> None:
        """
        Configures a single input scalar picker for the x-axis
        """
        self.input_pickers = [Select(
            title="x-axis",
            options=list(self.input_scalar_menu.keys()),
            sizing_mode='scale_width',
            value=self.input_scalars[self.initial_x_axis_scalar_ind].label,
        )]