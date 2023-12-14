# Use a slim version of Python 3.8
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the Python dependencies file to the container
COPY requirements.txt .

# Update the package list and install libgomp1
# This is required for lightgbm, a dependency of PyCaret
RUN apt-get update && apt-get install -y libgomp1

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application's code to the container
COPY . .

# Command to run the application
CMD ["python", "app.py"]
