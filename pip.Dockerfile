FROM python:3.12.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file
COPY requirements.txt /app/

# Install dependencies using pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . /app

# Expose the port your app is running on (optional)
EXPOSE 8000

# Command to run on container start
CMD ["python", "run.py"]
