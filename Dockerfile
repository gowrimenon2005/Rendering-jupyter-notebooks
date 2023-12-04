FROM continuumio/miniconda3:latest

# Set the working directory
WORKDIR /app

# Copy environment.yml to the container
COPY environment.yml .

# Create Conda environment and install voila
RUN conda env create -f environment.yml && \
    echo "conda activate my_environment" >> ~/.bashrc && \
    /bin/bash --login -c "conda init" && \
    /bin/bash --login -c "conda activate my_environment" && \
    conda install -n my_environment voila=0.5.5

# Copy the notebooks into the container
COPY notebooks /app/notebooks

# Expose the port that Voila will run on
EXPOSE 8866

# Set the default command to run when the container starts
CMD ["voila", "--port=8866", "--no-browser", "notebooks"]
