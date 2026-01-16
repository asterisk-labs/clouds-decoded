FROM continuumio/miniconda3:latest

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Create the environment:
RUN conda env create -f environment.yml



