"""
framework.py

This script contains the Framework class which initializes a GUI framework for visualizing simulations.
The class includes methods for creating the window, adding buttons, and displaying the simulation.
It aslo includes a Button class for creating graphical buttons in the GUI.

Dependencies:
----
- tkinter
- matplotlib

Usage:
----
1. Import the required libraries: tkinter, matplotlib.
2. Import the Framework class from this file.
3. Create an instance of the Framework class and use its methods to create a GUI for a simulation.

Example:
----
# Create a Framework instance
framework = Framework("Simulation", 800, 600)
Button(frame, text='Quit', command=self.quit_root, font=self.FONT)().pack(
            side=tk.TOP, pady=self.PADY, anchor=self.ANCHOR, padx=self.PADX
        )

# Add a button to the framework
framework.add_button("Start", start_simulation)

# Display the framework
framework.display()

"""

import tkinter as tk
import matplotlib.pyplot as plt

from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class Framework:
    """
    Framework class initializes a GUI framework for visualizing simulations.

    Parameters
    ----------
    title : str
        The title of the window.
    height : int
        The height of the window.
    width : int
        The width of the window.
    fullscreen : bool, optional (default=False)
        Whether the window should be displayed in fullscreen mode.

    Attributes
    ----------
    title : str
        The title of the window.
    height : int
        The height of the window.
    width : int
        The width of the window.
    fullscreen : bool
        Whether the window is displayed in fullscreen mode.
    plot_height : int
        The height of the matplotlib graph.
    plot_width : int
        The width of the matplotlib graph.
    FONT : tuple
        The font configuration for buttons.
    ANCHOR : str
        The anchor point for buttons.
    PADX : int
        The X-axis padding for buttons.
    PADY : int
        The Y-axis padding for buttons.
    root : tk.Tk
        The main tkinter root window.
    figure : matplotlib.figure.Figure
        The matplotlib figure object.
    axes : matplotlib.axes.Axes
        The matplotlib axis object.
    canvas : matplotlib.backends.backend_tkagg.FigureCanvasTkAgg
        The tkinter canvas widget.

    Methods
    -------
    create_graph_frame(container, propagate=False) -> tk.Frame
        Creates and returns a frame containing a matplotlib graph.
    create_button_frame(container, propagate=False) -> tk.Frame
        Creates and returns a frame containing buttons for interaction.
    create_root() -> None
        Creates the main tkinter root window with graph and button frames.
    run_root() -> None
        Starts the main loop for the tkinter root window.
    quit_root() -> None
        Quits the main loop for the tkinter root window.
    get_figure() -> matplotlib.figure.Figure
        Returns the matplotlib figure object.
    get_axis() -> matplotlib.axes.Axes
        Returns the matplotlib axis object.
    get_canvas() -> matplotlib.backends.backend_tkagg.FigureCanvasTkAgg
        Returns the tkinter canvas widget.
    set_plot_size(height, width) -> None
        Sets the matplot figure size.
    set_button_style(family="Helvetica", size=12, style=None, anchor="nw", padx=20, pady=3) -> None
        Sets the button style.

    Example
    -------
    framework = Framework("Simulation Framework", 800, 1200)
    framework.run_root()
    """

    def __init__(self, title: str, height: int, width: int, fullscreen: bool = False) -> None:
        """
        Initializes a new Framework instance.

        Parameters
        ----------
        - title (str): The title of the window.
        - height (int): The height of the window.
        - width (int): The width of the window.
        - fullscreen (bool): Whether the window should be displayed in fullscreen mode.
            Defaults to False.
        """
        self.title: str = title
        self.height: int = height
        self.width: int = width
        self.fullscreen = fullscreen

        self.plot_height: int = 10
        self.plot_width: int = 10

        self.FONT = ("Helvetica", 12)
        self.ANCHOR = "nw"
        self.PADX = 20
        self.PADY = 3

        self.create_root()

    def create_graph_frame(self, container: tk.Widget, propagate: bool = False) -> tk.Frame:
        """
        Create and return a frame containing a matplotlib graph.

        Parameters
        ----------
        - container (tk.Widget): The container where the frame will be placed.
        - propagate (bool): Whether the frame should propagate its size. Defaults to False.

        Returns
        -------
        - tk.Frame: The frame containing a matplotlib graph.
        """
        frame: tk.Frame = tk.Frame(container)
        frame.pack_propagate(propagate)

        self.figure: Figure
        self.axes: Axes
        self.figure, self.axes = plt.subplots(
            figsize=(self.plot_height, self.plot_width))
        self.canvas: FigureCanvasTkAgg = FigureCanvasTkAgg(
            self.figure, master=frame)
        canvas_widget: tk.Widget = self.canvas.get_tk_widget()
        canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        return frame

    def create_button_frame(self, container: tk.Widget, propagate: bool = False) -> tk.Frame:
        """
        Create and return a frame containing buttons for interaction.

        Parameters
        ----------
        - container (tk.Widget): The container where the frame will be placed.
        - propagate (bool) Whether the frame should propagate its size. Defaults to False.

        Returns
        -------
        - tk.Frame: The frame containing buttons for interaction.
        """

        frame: tk.Frame = tk.Frame(container)
        frame.pack_propagate(propagate)

        Button(frame, text='Quit', command=self.quit_root, font=self.FONT)().pack(
            side=tk.TOP, pady=self.PADY, anchor=self.ANCHOR, padx=self.PADX
        )
        Button(frame, text='Pause', command=None, font=self.FONT)().pack(
            side=tk.TOP, pady=self.PADY, anchor=self.ANCHOR, padx=self.PADX
        )

        return frame

    def create_root(self) -> None:
        """Create the main tkinter root window with graph and button frames."""
        self.root: tk.Tk = tk.Tk()
        self.root.title(self.title)
        self.root.geometry(f"{self.width}x{self.height}")
        FIRST_COLUMN_WEIGHT = 10

        self.root.columnconfigure(0, weight=FIRST_COLUMN_WEIGHT)
        self.root.columnconfigure(1, weight=1)

        plot_frame: tk.Frame = self.create_graph_frame(self.root)
        plot_frame.grid(column=0, row=0, sticky="nsew")

        button_frame: tk.Frame = self.create_button_frame(self.root)
        button_frame.grid(column=1, row=0, sticky="nsew")

        self.root.grid_columnconfigure(0, weight=FIRST_COLUMN_WEIGHT)
        self.root.grid_rowconfigure(0, weight=1)
        return

    def run_root(self) -> None:
        """Start the main loop for the tkintaer root window."""
        self.root.attributes("-fullscreen", self.fullscreen)
        self.root.mainloop()
        return

    def quit_root(self) -> None:
        """Quit the main loop for the tkintaer root window."""
        self.root.destroy()

    def get_figure(self) -> Figure:
        """Return the matplotlib figure object."""
        return self.figure

    def get_axis(self) -> Axes:
        """Return the matplotlib axis object."""
        return self.axes

    def get_canvas(self) -> FigureCanvasTkAgg:
        """Return the tkinter canvas widget."""
        return self.canvas

    def set_plot_size(self, height: int, width: int) -> None:
        """Set the matplot figure size."""
        self.plot_height = height
        self.plot_width = width
        return

    def set_button_style(self, family: str = "Helvetica",
                         size: int = 12, style=None, anchor: str = "nw",
                         padx: int = 20, pady: int = 3) -> None:
        """Set the button style."""
        self.FONT = (family, size) if style is None else (family, size, style)
        self.ANCHOR = anchor
        self.PADX = padx
        self.PADY = pady
        return


class Button:
    """
    Class representing a graphical button in a GUI application.

    Parameters
    ----------
    container : tk.Widget
        The container (frame, window, etc.) where the button will be placed.
    text : str
        The text to be displayed on the button.
    image : tk.PhotoImage or None, optional (default=None)
        An optional image to be displayed on the button.
    command : callable or None, optional (default=None)
        The function to be executed when the button is clicked.
        If None, a default message will be printed.
    font : tk.Font or None, optional (default=None)
        The font to be used for the button text.

    Attributes
    ----------
    container : tk.Widget
        The container where the button is placed.
    text : str
        The text displayed on the button.
    image : tk.PhotoImage or None
        The image displayed on the button.
    command : callable
        The function to be executed when the button is clicked.
    instance : tk.Button
        The Tkinter Button widget instance.
    font : tk.Font or None
        The font used for the button text.

    Class Attributes
    ----------------
    _instances : dict
        A dictionary containing all instances of the Button class with text as keys.

    Methods
    -------
    create_button()
        Creates the Tkinter Button widget.
    set_command(command)
        Sets a new command for the button.
    __call__(*args, **kwargs)
        Returns the Tkinter Button widget instance.

    Class Methods
    -------------
    init_button_command()
        A default method called when no command is provided.
    get_all_instances() -> dict
        Returns a dictionary containing all instances of the Button class.

    Example
    -------
    button = Button(container, "Click me", command=my_function, font=my_font)
    """
    _instances = {}

    @classmethod
    def init_button_command(cls) -> None:
        """Default method called when no command is provided."""
        print("No command found!")
        return

    @classmethod
    def get_all_instances(cls) -> None:
        """Return a dictionary containing all instances of the Button class."""
        return cls._instances

    def __init__(self, container: tk.Widget, text: str, image: tk.PhotoImage = None,
                 command: callable = None, font: dict = None) -> None:
        """
        Initializes a new Button instance.

        Parameters
        ----------
        - container (tk.Widget) The container where the button will be placed.
        text (str): The text to be displayed on the button.
        image (tk.PhotoImage): An optional image to be displayed on the button. Defaults to None.
        command (callable) The function to be executed when the button is clicked.
            If None, a default message will be printed.
        font (Dict): The font to be used for the button text. Defaults to None.
        """
        self.container: tk.Widget = container
        self.text: str = text
        self.image: tk.PhotoImage = image
        self.command: callable = command if command is not None else self.init_button_command
        self.instance = None
        self.font: dict = font

        self.create_button()
        Button._instances[self.text] = self

    def create_button(self) -> None:
        """Creates the Tkinter Button widget."""
        self.instance = tk.Button(master=self.container, text=self.text,
                                  image=self.image, command=self.command, font=self.font)
        return

    def set_command(self, command) -> None:
        """Sets a new command for the button."""
        self.command = command
        self.instance.config(command=self.command)
        return

    def __call__(self, *args, **kwargs) -> tk.Button:
        """Returns the Tkinter Button widget instance."""
        return self.instance
