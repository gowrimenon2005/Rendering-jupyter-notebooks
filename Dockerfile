FROM continuumio/miniconda3:latest

# Set the working directory
WORKDIR /app

# Copy environment.yml to the container
COPY environment.yml .

# Create Conda environment
RUN conda env create -f environment.yml

# Activate the Conda environment
RUN echo "conda activate my_environment" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

# Install voila
RUN conda install -n my_environment voila=0.5.5

# Copy the notebooks into the container
COPY notebooks /app/notebooks

# Expose the port that Voila will run on
EXPOSE 8866

# Command to run Voila
CMD ["voila", "--port=8866", "--no-browser", "notebooks"]
