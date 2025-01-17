from modules.config import ConfigManager, config_manager
from modules.logly import logly
from webui import FineTuneHandler, FineTuneUI
from modules.arguments import parse_arguments


def main(args):
    # Initialize the config manager and load the config
    config = config_manager.load_config()

    # Check if debugging is enabled, either through command line or config file
    debug_mode = args.debug if args.debug is not None else config["project_host"]["debug"]

    # Check for ssr_mode, either through command line or config file
    ssr_mode = args.ssr_mode if args.ssr_mode is not None else config["project_host"]["ssr_mode"]

    # Check for logging enabled or not
    logging_enabled = args.logging if args.logging is not None and debug_mode else config["settings"]["logging_enabled"]

    if debug_mode:
        logly.setLevel("DEBUG")  # Set log level to DEBUG if debugging is enabled
        logly.debug("Debug mode enabled.")  # Log a debug message
    logly.info("Starting Fine-Tune Web UI.")
    handler = FineTuneHandler()
    ui = FineTuneUI(handler)

    if logging_enabled and debug_mode:
        logly.info(f"Starting Fine-Tune Web UI with the following options:")
        logly.info(f"Host: {args.host or config['project_host']['host']}")
        logly.info(f"Port: {args.port or config['project_host']['port']}")
        logly.info(f"Debug: {debug_mode}")

    # Create the Gradio UI and launch the application
    app = ui.create_ui()
    app.launch(server_name=args.host or config["project_host"]["host"],
               server_port=args.port or config["project_host"]["port"],
               debug=debug_mode, ssr_mode=ssr_mode)


if __name__ == "__main__":
    # Parse arguments using the separate arguments file
    args = parse_arguments()
    main(args)
