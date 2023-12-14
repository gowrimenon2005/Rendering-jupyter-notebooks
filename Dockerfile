FROM continuumio/miniconda3:latest

# Set the working directory
WORKDIR /app

# Copy environment.yml to the container
COPY environment.yml .

# Create Conda environment and install voila
RUN conda env create -f environment.yml

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "my_environment", "/bin/bash", "-c"]

# Copy the notebooks into the container
COPY notebooks /app/notebooks

# Expose the port that Voila will run on
EXPOSE 8866

# Set the default command to run when the container starts
CMD ["voila", "--port=8866", "--no-browser", "notebooks"]
