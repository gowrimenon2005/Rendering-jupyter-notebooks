FROM python:3.10

# Set the working directory
WORKDIR /testproject

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install any dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Voila notebooks into the container
COPY notebooks /testproject/notebooks

# Expose the port that Voila will run on
EXPOSE 8866

# Command to run Voila
CMD ["voila", "--port=8866", "--no-browser", "notebooks"]
