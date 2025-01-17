from webui import FineTuneHandler, FineTuneUI
from modules.arguments import parse_arguments

def main(args):
    # Initialize the handler and UI
    handler = FineTuneHandler()
    ui = FineTuneUI(handler)

    # Print received arguments (for demonstration or debugging)
    print(f"Starting Fine-Tune Web UI with the following options:")
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Debug: {args.debug}")

    # Create the Gradio UI and launch the application
    app = ui.create_ui()
    app.launch(server_name=args.host, server_port=args.port, debug=args.debug)

if __name__ == "__main__":
    # Parse arguments using the separate arguments file
    args = parse_arguments()
    main(args)
