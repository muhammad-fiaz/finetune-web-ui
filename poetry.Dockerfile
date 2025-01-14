FROM python:3.12.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file (or Poetry's lock file if using Poetry)
COPY pyproject.toml poetry.lock /app/

# Install Poetry if you're using it
RUN pip install poetry

# Install dependencies using Poetry (or pip if you have a requirements.txt)
RUN poetry install --no-root

# Copy the rest of your application code
COPY . /app

# Expose the port your app is running on (optional)
EXPOSE 8000

# Command to run on container start
CMD ["poetry", "run", "python", "run.py"]
