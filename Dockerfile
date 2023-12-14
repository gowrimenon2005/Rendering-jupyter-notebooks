FROM python:3.8-slim

# Set the working directory
WORKDIR /app

# Copy environment.yml to the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the notebooks into the container
COPY notebooks /app/notebooks

# Expose the port that Voila will run on
EXPOSE 8866

# Set the default command to run when the container starts
CMD ["voila", "--port=8866", "--no-browser", "notebooks"]
