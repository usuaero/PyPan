class MachInclinedError(Exception):
    """An exception thrown when a panel is Mach inclined."""

    def __init__(self):
        super().__init__("This panel is Mach inclined.")