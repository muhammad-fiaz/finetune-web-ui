# Docker Setup for Finetune Web UI

This guide provides instructions for building and running the **Finetune Web UI** project using Docker. The project enables a user-friendly interface for training and deploying machine learning models, specifically designed for fine-tuning pre-trained models such as Llama, GPT, etc.

## Prerequisites

Before proceeding, make sure you have the following installed on your machine:

- **Docker**: Install Docker from [here](https://docs.docker.com/get-docker/).
- **Docker Compose** (optional): Useful if you want to run multi-container setups.

## Dockerfiles

The project includes two Dockerfiles for building Docker images based on different dependency management systems:

- **`pip.Dockerfile`**: Dockerfile for pip-based dependency management (using `requirements.txt`).
- **`poetry.Dockerfile`**: Dockerfile for Poetry-based dependency management (using `pyproject.toml` and `poetry.lock`).

You can choose the appropriate Dockerfile depending on your preference for managing dependencies.

## Building Docker Images

### 1. **For pip-based Project**:

If you're using **pip** and have a `requirements.txt` file, follow these steps:

1. **Build the Docker image**:
   ```bash
   docker build -t finetune-web-ui-pip -f pip.Dockerfile .
   ```

2. **Run the Docker container**:
   ```bash
   docker run -p 7860:7860 finetune-web-ui-pip
   ```

This will start the **Finetune Web UI** and expose it on port `7860` by default.

### 2. **For Poetry-based Project**:

If you're using **Poetry** for dependency management, and have `pyproject.toml` and `poetry.lock`, follow these steps:

1. **Build the Docker image**:
   ```bash
   docker build -t finetune-web-ui-poetry -f poetry.Dockerfile .
   ```

2. **Run the Docker container**:
   ```bash
   docker run -p 7860:7860 finetune-web-ui-poetry
   ```

This will also start the **Finetune Web UI** and expose it on port `7860`.

### 3. **With Docker Compose** (Optional):

If you'd like to use Docker Compose for managing the application, you can create a `docker-compose.yml` file in the root of the project. Here’s an example:

```yaml
version: '3'

services:
  finetune-web-ui:
    build:
      context: .
      dockerfile: poetry.Dockerfile  # Change this to 'pip.Dockerfile' for pip-based setup
    ports:
      - "7860:7860"
    volumes:
      - .:/app
    environment:
      - ENV=production
```

To build and run the container with Docker Compose:

```bash
docker-compose up --build
```

This will automatically build the Docker image and start the container.

## `.dockerignore`

Both Dockerfiles use a `.dockerignore` file to exclude unnecessary files from being copied into the Docker image. Here’s an example `.dockerignore` file:

```plaintext
__pycache__
*.pyc
*.pyo
*.git
*.gitignore
```

This ensures that only the relevant files are included in the Docker image, reducing its size.

## Running the Application

Once the Docker container is running, the **Finetune Web UI** should be accessible at:

```
http://localhost:7860
```

You can upload datasets, fine-tune models, and monitor training progress through the web interface.

## Project Structure

Here’s a summary of the project structure:

```bash
finetune-web-ui/
├── run.py                # Main application script to run the UI
├── pip.Dockerfile        # Dockerfile for pip-based dependency management
├── poetry.Dockerfile     # Dockerfile for Poetry-based dependency management
├── requirements.txt      # List of dependencies (if pip isn't used)
├── pyproject.toml        # Poetry project configuration (if using Poetry)
├── poetry.lock           # Poetry lock file (if using Poetry)
├── .dockerignore         # Docker ignore file
├── README.md             # Project documentation
├── docker_README.md      # Docker usage documentation
├── logs/                 # Folder for storing logs
└── configs/              # Directory for storing configuration files

```

## Contributing

We welcome contributions to **Finetune Web UI**! If you have suggestions, bug fixes, or new features, feel free to submit a pull request.

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -am 'Add new feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Open a pull request.

## License

This project is licensed under the AGPL-3.0 License. See the [LICENSE](LICENSE) file for more details.

---

## Acknowledgments

- [Gradio](https://gradio.app/) - For building the UI components.
- [Hugging Face](https://huggingface.co/) - For pre-trained models and transformers.
- [PyTorch](https://pytorch.org/) - For the deep learning framework.
- [Datasets](https://huggingface.co/datasets) - For providing datasets for training.
- [Unsloth](https://github.com/unsloth) - For **FastLanguageModel** and other contributions to the training pipeline.
