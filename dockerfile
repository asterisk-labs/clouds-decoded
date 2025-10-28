# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Set the default command to execute the Python script with the provided arguments
CMD ["python", "./src/main.py", "--config", "/app/config.yaml", "--scene_dir", "/mnt/raid/data/s2get/Sentinel-2/MSI/L1C/2025/08/01/S2C_MSIL1C_20250801T232231_N0511_R044_T58NFN_20250802T005448.SAFE", "--plot", "--save", "--log"]docker run -v /home/paul/cloud-height-prototype/config.yaml:/app/config.yaml -v /mnt/raid/data:/mnt/raid/data <image_name>
