FROM python:3.12.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file (or uv's lock file if using uv)
COPY pyproject.toml /app/

# Install uv if you're using it
RUN pip install uv

# Install dependencies using uv (or pip if you have a requirements.txt)
RUN uv lock

# Copy the rest of your application code
COPY . /app

# Expose the port your app is running on (optional)
EXPOSE 8000

# Command to run on container start
CMD ["uv", "run", "run.py"]
