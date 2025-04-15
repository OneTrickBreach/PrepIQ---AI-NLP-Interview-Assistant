# PrepIQ - AI-NLP Interview Assistant

An AI-powered platform to generate interview questions, evaluate candidate answers, and provide detailed feedback for technical and behavioral interviews across multiple software and data roles.

---

## Objectives

- **Automate interview preparation** with role-specific question generation.
- **Evaluate candidate answers** using custom-trained NLP models.
- **Provide actionable feedback** highlighting strengths and areas for improvement.
- **Support multiple roles** including Software Engineer, Data Scientist, DevOps, QA, Mobile Developer, and more.
- **Enable speech-based answers** with integrated speech-to-text (using OpenAI Whisper).
- **Facilitate model training and improvement** with extensible data pipelines.

---

## Expected Outcomes

- A web-based tool for interview practice with realistic, role-specific questions.
- Automated scoring and feedback on candidate answers.
- Support for both text and audio answers.
- Extensible backend for adding new roles, questions, and improving models.

---

## Directory Structure

```
.
├── src/                    # Backend Python code (FastAPI server, models, feedback)
│   ├── main.py             # FastAPI app entry point
│   ├── speech_to_text.py   # OpenAI Whisper integration
│   ├── data_generation.py  # Synthetic data generation
│   ├── models/             # Custom model definitions
│   ├── feedback/           # Feedback generation logic
│   ├── integration/        # Integration pipelines
│   ├── schemas/            # Pydantic schemas
│   ├── training/           # Training utilities
│   └── api/                # API submodules
│
├── scripts/                # Python scripts for data prep, training, evaluation
│   ├── train_custom_evaluator.py  # Train/fine-tune the custom evaluation model
│   ├── generate_dataset.py        # Generate synthetic dataset
│   ├── organize_data.py           # Organize data into training format
│   ├── evaluate_models.py         # Evaluate trained models
│   └── ...                        # Other utilities
│
├── frontend/               # React frontend app
│   ├── package.json
│   ├── src/
│   │   ├── App.js
│   │   ├── api/            # API client code
│   │   ├── components/     # UI components
│   │   └── pages/          # App pages (Interview, Results, etc.)
│   └── public/
│
├── organized_data/         # Processed data for training
├── models/                 # Saved model checkpoints
├── requirements.txt        # Python dependencies
├── .gitignore              # Git ignore rules
├── README.md               # This file

```

## Setup Instructions

### Prerequisites

Before you begin, ensure you have the following installed on your system:

1.  **Python:** Version 3.8 or higher.
2.  **Git:** For cloning the repository.
3.  **Node.js and npm:** Required for the frontend application. Download from [https://nodejs.org/](https://nodejs.org/).
4.  **FFmpeg:** Required by the Whisper library for audio processing.
    *   **Windows:** Install using Chocolatey (`choco install ffmpeg`) or download from the [FFmpeg website](https://ffmpeg.org/download.html) and add it to your system's PATH.
    *   **macOS:** Install using Homebrew (`brew install ffmpeg`).
    *   **Linux:** Install using your package manager (e.g., `sudo apt update && sudo apt install ffmpeg` on Debian/Ubuntu).

### 1. Clone the Repository

```bash
git clone https://github.com/OneTrickBreach/PrepIQ---AI-NLP-Interview-Assistant.git
cd PrepIQ---AI-NLP-Interview-Assistant
```

### 2. Python Backend Setup

- Navigate to the project root directory if you aren't already there.
- Create and activate a Python virtual environment (recommended):

  ```bash
  # Create the environment (only needs to be done once)
  python -m venv .venv 

  # Activate the environment (do this every time you work on the project)
  # Windows (PowerShell):
  .\.venv\Scripts\Activate.ps1
  # Windows (Command Prompt):
  # .\.venv\Scripts\activate.bat
  # macOS/Linux:
  # source .venv/bin/activate 
  ```

- Install Python dependencies:

  ```bash
  # Ensure your virtual environment is active first!
  pip install -r requirements.txt
  ```
  *(Note: This will install PyTorch and Whisper. The first time Whisper runs, it may download the model weights, which can take some time and disk space.)*

### 3. Download the Trained Evaluation Model

The custom-trained evaluation model (`.pt` file) is required for the answer evaluation feature but is **not included** in this repository due to size limits.

- Download the latest trained model checkpoint from this link:
  **[https://drive.google.com/file/d/1puq4Luf4aBJQUyw6FHLV7jU5yAUOgzX_/view?usp=sharing]** *(Link provided by user)*

- After downloading, place the `.pt` file (e.g., `custom_evaluator_best.pt`) inside the `models/` directory in the project root. The backend will automatically load the latest model file from this directory when started.

### 4. Frontend Setup

- Navigate to the frontend directory:

  ```bash
  cd frontend
  ```

- Install Node.js dependencies:

  ```bash
  npm install
  ```

- Go back to the project root directory:

  ```bash
  cd .. 
  ```

### 5. Generating Sample Data (Optional)

If you want to generate synthetic sample data for testing or retraining:

```bash
# Ensure your Python virtual environment is active
python scripts/generate_dataset.py
python scripts/organize_data.py
```

> **Note:** This synthetic data is very basic. For real training, replace or augment it with real, annotated examples.

### 6. Running the Application

You need to run the backend and frontend servers simultaneously in separate terminals.

- **Terminal 1: Run Backend Server**
  - Make sure you are in the project root directory (`PrepIQ---AI-NLP-Interview-Assistant`).
  - Activate your Python virtual environment (`.\.venv\Scripts\Activate.ps1` or `source .venv/bin/activate`).
  - Start the FastAPI server:

    ```bash
    python -m src.main
    ```

- **Terminal 2: Run Frontend Server**
  - Open a new terminal.
  - Navigate to the `frontend` directory: `cd frontend`
  - Start the React development server:

    ```bash
    npm start
    ```

- Once both servers are running, the application should be available in your web browser at `http://localhost:3000`.

---

## Training the Evaluation Model (Optional)

- To train from scratch:

```bash
# Ensure virtual environment is active
python scripts/train_custom_evaluator.py --data_dir organized_data/ --output_dir models/ --num_epochs 20 --batch_size 8
```

- To continue training from a checkpoint:

```bash
# Ensure virtual environment is active
python scripts/train_custom_evaluator.py --data_dir organized_data/ --output_dir models/ --num_epochs 10 --batch_size 8 --load_checkpoint "models/custom_evaluator_best.pt"
```

---

## Notes

- The `.gitignore` excludes large data/model directories and environment files.
- Placeholder files are included in excluded directories to preserve structure.
- You can customize roles, questions, and feedback templates by editing the data files or generation scripts.
- Speech-to-text is now handled locally using Whisper. Performance depends on your hardware and the chosen Whisper model size (default is "base").

---
