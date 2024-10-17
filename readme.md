# FastAPI Gradio Project

This project is a FastAPI application integrated with Gradio for a system context conversation agent using the Groq model.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/repository-name.git
    cd repository-name
    ```

2. Create a Conda environment:
    ```bash
    conda create --name myenv python=3.8
    conda activate myenv
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

4. Set up your environment variables by creating a `.env` file in the project root with the following content:
    ```plaintext
    API_KEY=your_api_key_here
    ```

## Running the Application

To run the application, use the following command:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
