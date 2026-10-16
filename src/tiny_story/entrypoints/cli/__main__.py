"""Run the dataset command line."""

from tiny_story.entrypoints.cli.app import app
from tiny_story.entrypoints.cli.clean import clean
from tiny_story.entrypoints.cli.download import download
from tiny_story.entrypoints.cli.normalize import normalize


def main() -> None:
    """Run the registered dataset commands.

    Returns:
        None.
    """
    registered = {clean.__name__, download.__name__, normalize.__name__}
    if registered != {"clean", "download", "normalize"}:
        message = "Dataset commands were not registered"
        raise RuntimeError(message)
    app()


if __name__ == "__main__":
    main()
