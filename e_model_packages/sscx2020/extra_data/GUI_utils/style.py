"""Style-related functions."""
import matplotlib as mpl


def get_style_cst():
    """Returns dict containing style vars such as colors."""
    style_dict = {}
    # font & width. has to be an attribute to be accessible.
    # somehow, entry font & width cannot be configurated with style.
    style_dict["base_font"] = "Helvetica 10"
    style_dict["entry_width"] = 8

    # BBP colors
    style_dict["light_blue"] = "#15D3FF"
    style_dict["blue"] = "#0B83CD"
    style_dict["deep_blue"] = "#050A58"
    style_dict["light_grey"] = "#F2F2F2"
    style_dict["grey"] = "#888888"
    style_dict["deep_grey"] = "#333333"
    style_dict["white"] = "#FFFFFF"

    return style_dict


def set_matplotlib_style():
    """Configure ticks & labels size."""
    mpl.rcParams["lines.color"] = get_style_cst()["blue"]
    mpl.rcParams["axes.labelsize"] = 8
    mpl.rcParams["xtick.labelsize"] = 8
    mpl.rcParams["ytick.labelsize"] = 8


def define_style(style):
    """Define the style for ttk objects."""
    style_dict = get_style_cst()

    style.configure(
        "TButton",
        background=style_dict["white"],
        font=style_dict["base_font"],
    )

    style.configure(
        "ControlSimul.TButton",
        padding=6,
        relief="solid",
        background=style_dict["white"],
        foreground=style_dict["deep_blue"],
        font="Helvetica 16 bold",
        borderwidth=2,
        highlightbackground=style_dict["deep_blue"],  # border color?
    )

    style.map(
        "ControlSimul.TButton",
        foreground=[
            ("pressed", "!disabled", style_dict["blue"]),
            ("disabled", style_dict["grey"]),
        ],
    )

    style.configure("TFrame", background=style_dict["white"])
    style.configure(
        "Boxed.TFrame",
        background=style_dict["white"],
        relief="solid",
        bordercolor=style_dict["deep_blue"],
        borderwidth=4,
    )

    style.configure(
        "TRadiobutton",
        background=style_dict["white"],
        relief="flat",
        cursor="dot",
        borderwidth=0,
        selectcolor=style_dict["blue"],
        font=style_dict["base_font"],
    )

    style.map(
        "TRadiobutton",
        foreground=[
            ("selected", style_dict["blue"]),
            ("!selected", style_dict["deep_blue"]),
        ],
    )

    style.configure(
        "TLabel",
        foreground=style_dict["deep_blue"],
        background=style_dict["white"],
        font=style_dict["base_font"],
    )

    style.configure(
        "TEntry",
        foreground="black",
        background=style_dict["white"],
    )

    style.map(
        "TEntry",
        highlightcolor=[("focus", style_dict["blue"])],
        bordercolor=[("focus", style_dict["blue"])],
    )

    style.configure(
        "TCheckbutton",
        foreground=style_dict["deep_blue"],
        background=style_dict["white"],
        font=style_dict["base_font"],
    )

    style.map(
        "TCombobox",
        fieldbackground=[("!disabled", style_dict["white"])],
        background=[("!disabled", style_dict["white"])],
    )
