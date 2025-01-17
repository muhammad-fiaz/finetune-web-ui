import argparse

def parse_arguments():
    """
    Parses command-line arguments for the Fine-Tune Web UI launcher.
    Returns:
        args: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Launcher for Fine-Tune Web UI")
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address to run the web UI (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the web UI (default: 7860)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run the app in debug mode"
    )
    return parser.parse_args()
