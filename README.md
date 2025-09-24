# ClaimClarity AI: AI-Powered Insurance Claim Processing

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

A comprehensive solution that leverages AI to automate insurance claim analysis and decision-making, aiming to solve the inefficiency in the insurance claims process.

## 📋 Table of Contents

- [About The Project](#about-the-project)
- [Key Features](#-key-features)
- [Tech Stack](#-tech-stack)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

## 🌟 About The Project

In the fast-paced world of financial services, there is a constant need for innovation. ClaimClarity AI introduces a platform designed to leverage AI to automate insurance claim processing. It provides a seamless, secure, and user-friendly experience to tackle critical challenges in the FinTech industry by using a multi-agent RAG pipeline to analyze claims and provide decisions with justifications.

## ✨ Key Features

* **Automated Claim Processing:** Uses a multi-agent RAG pipeline to analyze and process insurance claims automatically.
* **Intuitive Dashboard:** A clean and interactive user interface built with Streamlit to visualize claim processing results and statistics.
* **Real-time Processing:** Provides instant claim analysis and decisions.
* **Data Security:** Best practices for ensuring user data is safe and encrypted.
* **Scalable Architecture:** Built with a modular design using FastAPI for the backend, allowing for easy future expansion.
* **Voice-based Support:** Includes a voice assistant for handling natural language queries.
* **FAQ Pipeline:** A separate pipeline to handle general policy questions.

## 🛠️ Tech Stack

This project is built using a modern and robust technology stack:

* **Backend:** Python with FastAPI and LangGraph
* **Frontend:** Streamlit
* **Database:** Pinecone, Upstash
* **Deployment:** Render

## 🚀 Getting Started

Follow these instructions to get a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Make sure you have the following software installed on your system:
* Python 3.8+
* pip (Python package installer)
* Git

### Installation

1.  **Clone the repository:**
    ```sh
    git clone [https://github.com/Shansgupta/Bajaj-hackathon.git](https://github.com/Shansgupta/Bajaj-hackathon.git)
    ```
2.  **Navigate to the project directory:**
    ```sh
    cd Bajaj-hackathon
    ```
3.  **Create and activate a virtual environment:**
    ```sh
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```
4.  **Install the required dependencies:**
    ```sh
    pip install -r requirements.txt
    ```
5.  **Set up environment variables:**
    * Create a `.env` file in the root directory.
    * Add the following environment variables to the `.env` file:
        ```
        OPENAI_API_KEY=your_openai_api_key
        PINECONE_API_KEY=your_pinecone_api_key
        SERPAPI_KEY=your_serpapi_key
        UPSTASH_VECTOR_URL=your_upstash_vector_url
        UPSTASH_VECTOR_TOKEN=your_upstash_vector_token
        ```

## 🏃 Usage

1.  **Run the Streamlit application:**
    ```sh
    streamlit run app.py
    ```
2.  **Run the FastAPI server:**
    ```sh
    uvicorn api.main:app --reload
    ```
3.  **Open your web browser** and navigate to the Streamlit URL (usually `http://localhost:8501`) to use the claim processing interface.
4.  The API will be available at `http://127.0.0.1:8000`.

## 📁 Project Structure

Here is an overview of the project's directory structure: