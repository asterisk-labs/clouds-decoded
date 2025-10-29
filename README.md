# Cloud Height Prototype

This project is a prototype for calculating cloud heights using Sentinel2 parallax effect https://isprs-annals.copernicus.org/articles/V-1-2021/17/2021/isprs-annals-V-1-2021-17-2021.pdf

## Installation
Either install requirements via conda or use the provided Dockerfile.

1. Clone the repository and navigate to the project directory:
    ```bash
    git clone https://github.com/your-username/cloud-height-prototype.git
    cd cloud-height-prototype
    ```

2. Install the required dependencies with conda:
    ```bash
    conda env create -f environment.yaml
    conda activate cloud-height-prototype
    ```

## Usage

1. Modify the `config.yaml` file to set your parameters (pay attention to OUTPUT_DIR)
2. Prepare your Sentinel2 scene directory (.SAFE) using S2get https://github.com/asterisk-labs/s2get
3. Run the main script:
    ```bash
    python main.py 
    --scene_dir /path/to/your/sentinel2/scene
    --config ./config.yaml # or /path/to/your/config.yaml 
    --plot
    --save
    --log
    ```
    
    Or using Docker:
    ```bash
    docker build -t cloud-height-prototype .
    docker compose up -d
    docker compose run cloud-height-prototype 
        --scene_dir /path/to/your/sentinel2/scene
        --config /app/config.yaml 
        --plot
        --save
        --log
    ```

