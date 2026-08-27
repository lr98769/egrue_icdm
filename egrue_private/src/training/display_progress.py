import matplotlib.pyplot as plt
from math import ceil
from plotly.graph_objects import FigureWidget
from plotly.subplots import make_subplots
from IPython.display import display

plotly_colours  = [
    '#1f77b4',  # muted blue
    '#ff7f0e',  # safety orange
    '#2ca02c',  # cooked asparagus green
    '#d62728',  # brick red
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf' # blue-teal
]

class LiveTrainingProgress:
    def __init__(self, metric_names, ncols= 3, x_axis_name="Epoch", splits=["Train", "Valid"]):
        self.x_axis_name = x_axis_name
        self.metrics = metric_names
        self.n_metrics = len(self.metrics)
        self.splits = splits
        self.n_splits = len(self.splits)
        # Compute Cols
        self.ncols = min(ncols, self.n_metrics)
        self.nrows = ceil(self.n_metrics/self.ncols)
        # Create Figure
        self.figure = FigureWidget(make_subplots(
            rows=self.nrows, cols=self.ncols, horizontal_spacing = 0.1)) # 
        self.scatters = {}
        assert self.n_metrics < len(plotly_colours)
        for i_metric, metric_name in enumerate(self.metrics):
            i_row, i_col = i_metric//self.ncols+1, i_metric%self.ncols+1
            for i_split, split in enumerate(self.splits):
                self.figure.add_scatter(
                    y=[], row=i_row, col=i_col, name=split.capitalize(),
                    line=dict(color=plotly_colours[i_split])
                )
                if split not in self.scatters:
                    self.scatters[split] = {}
                # Add scatter plot to dictionary
                self.scatters[split][metric_name] = self.figure.data[i_metric*self.n_splits+i_split]
                # Only show legend for first plot
                self.scatters[split][metric_name]['showlegend'] = i_metric == 0
            # Set axis labels
            self.figure['layout'][f'xaxis{i_metric+1}']['title'] = self.x_axis_name.capitalize()
            self.figure['layout'][f'yaxis{i_metric+1}']['title'] = metric_name.capitalize()
        self.figure.update_layout(
            margin={'t':10,'l':0,'b':0,'r':0},
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right",x=1)
        )
        display(self.figure)
    
    def update_figure(self, value_dict):
        for split, metric_dict in value_dict.items():
            for metric, metric_values in metric_dict.items():
                self.scatters[split][metric].y = metric_values

def plot_history(history, split_list, metric_list, max_cols = 2, fp_history=None, show=True):
    plot_width, plot_height = 3, 2
    # Calculate number of plots
    num_metrics = len(metric_list)
    if num_metrics <= max_cols:
        num_rows = 1
        num_cols = num_metrics
    else:
        num_rows = ceil(num_metrics/max_cols)
        num_cols = max_cols
    # Get epoch vector
    num_epochs = len(history[split_list[0]][metric_list[0]])
    epochs = [i for i in range(num_epochs)]
    # Make axes
    fig, axes = plt.subplots(
        num_rows, num_cols, figsize=(num_cols*plot_width, num_rows*plot_height), dpi=300)
    if num_rows > 1:
        axes = axes.flatten()
    elif num_metrics == 1:
        axes = [axes]
    # Plot metrics
    for i, metric in enumerate(metric_list):
        for split in split_list:
            axes[i].plot(epochs, history[split][metric], label=split.capitalize() if i == 0 else None)
        axes[i].set_ylabel(metric.capitalize())
        axes[i].set_xlabel("Epochs")
    # Remove additional plots
    if num_rows*num_cols > num_metrics:
        for j in range(i+1,num_rows*num_cols):
            axes[j].axis('off')
    # Add legend
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0), ncol=len(split_list))
    plt.tight_layout()
    
    # Save plots or show plot
    if fp_history:
        plt.savefig(fp_history, bbox_inches="tight")
        
    if show:
        plt.show()
    else:
        plt.close(fig)

def print_split_epoch_metrics(splitname, metric_dict):
    print(f"- {splitname.capitalize()}: ", end="")
    last_metrics_idx = len(metric_dict)-1
    for i, (metric, val) in enumerate(metric_dict.items()):
        print(f"{metric.capitalize()}: {val:.5f}", end=", " if i != last_metrics_idx else "\n")