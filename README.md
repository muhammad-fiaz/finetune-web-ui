<h1 align="center">Finetune Web UI</h1>

**Finetune Web UI** is a user-friendly interface for training and deploying machine learning models, specifically designed for fine-tuning pre-trained models like Llama, GPT, etc. The project aims to streamline the process of model fine-tuning, allowing data scientists and engineers to easily manage their training configurations, datasets, and models through a clean and intuitive web interface.

---
> **Note:** This project is currently under development and not yet ready for production use. Feel free to contribute to the project by submitting pull requests, reporting issues, or suggesting new features.

## Features

- **Easy Dataset Upload:** Upload your datasets directly from a URL or local files.
- **Custom Model Fine-tuning:** Select pre-trained models like Llama, GPT, etc., and fine-tune them with your custom datasets.
- **Hyperparameter Tuning:** Configure training parameters like batch size, learning rate, etc.
- **Real-time Monitoring:** Track the progress of your fine-tuning in real time.
- **Simplified deployment:** Once the model is fine-tuned, deploy it instantly for inference.
- **Model Management:** Save and manage your fine-tuned models for future use.
- **User-friendly Interface:** Clean and intuitive UI for easy navigation and interaction.
- **Open Source:** Free and open-source project for the community.
- **Customizable:** Extend and modify the project to suit your needs.
- **Community Support:** Active community and contributors for help and guidance.
- **Cross-platform:** Works on Windows, macOS, and Linux.
- **Scalable:** Can be deployed on cloud services for large-scale training.

---

## Requirements

### Prerequisites:
- Python 3.12 or above
- Poetry (for managing dependencies)
- Virtual environment recommended

---

## Installation

Follow these steps to install and run the `finetune-web-ui` locally.

### Step 1: Clone the Repository

```bash
git clone https://github.com/muhammad-fiaz/finetune-web-ui.git
cd finetune-web-ui
```
### Step 2.1: Install Dependencies with Pip

If you prefer using `pip` to install dependencies, run:

```bash
pip install -r requirements.txt
```

This will install all the necessary dependencies.


### Step 2.2: Install Dependencies with Poetry

If you don't have Poetry installed, follow the [Poetry installation guide](https://python-poetry.org/docs/#installation).

Once Poetry is installed, run:

```bash
poetry install
```

This will create a virtual environment and install all the necessary dependencies.

### Step 3: Activate Virtual Environment

Activate the environment:

```bash
poetry shell
```

Alternatively, you can activate the virtual environment using the `.venv` folder if you're using one:

```bash
source .venv/bin/activate  # For Linux/MacOS
.\.venv\Scripts\activate  # For Windows
```

### Step 4: Run the Application

Run the following command to start the web interface:

```bash
python run.py
```

This will launch the web UI, typically accessible at `http://localhost:7860`.

---

## Usage

1. **Dataset Configuration**: Upload a dataset by providing a URL or select a local file. Your data should be in a format supported by the training script.
2. **Model Selection**: Choose from available pre-trained models like Llama, GPT, etc.
3. **Training Configuration**: Adjust hyperparameters like learning rate, batch size, and training epochs.
4. **Training & Monitoring**: Start the fine-tuning process and monitor the training progress in real time.
5. **Model Deployment**: After fine-tuning, deploy your model directly through the interface for inference.

---

## Project Structure

```bash
finetune-web-ui/
├── run.py                # Main application script to run the UI
├── requirements.txt      # List of dependencies (if Poetry isn't used)
├── .venv/                # Virtual environment directory
├── README.md             # Project documentation
├── logs/                 # Folder for storing logs
└── configs/              # Directory for storing configuration files

```

---

## Contributing

We welcome contributions to `finetune-web-ui`! If you have suggestions, bug fixes, or new features, feel free to submit a pull request.

1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Make your changes
4. Commit your changes (`git commit -am 'Add new feature'`)
5. Push to the branch (`git push origin feature-branch`)
6. Open a pull request

---

## License

This project is licensed under the AGPL-3.0 License. See the [LICENSE](LICENSE) file for more details.


## Acknowledgments

- [Gradio](https://gradio.app/) - For building the UI components
- [Hugging Face](https://huggingface.co/) - For pre-trained models and transformers
- [PyTorch](https://pytorch.org/) - For deep learning framework
- [Datasets](https://huggingface.co/datasets) - For providing datasets for training
- [Unsloth](https://github.com/unsloth) - For **FastLanguageModel** and other contributions to the training pipeline



