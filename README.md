# PrepIQ - AI-NLP Interview Assistant

An AI-powered platform to generate interview questions, evaluate candidate answers, and provide detailed feedback for technical and behavioral interviews across multiple software and data roles.

---

## Objectives

- **Automate interview preparation** with role-specific question generation.
- **Evaluate candidate answers** using custom-trained NLP models.
- **Provide actionable feedback** highlighting strengths and areas for improvement.
- **Support multiple roles** including Software Engineer, Data Scientist, DevOps, QA, Mobile Developer, and more.
- **Enable speech-based answers** with integrated speech-to-text.
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
│   ├── speech_to_text.py   # Google Speech-to-Text integration
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
├── data/                   # Raw data files (excluded from Git)
├── organized_data/         # Processed data for training (excluded from Git)
├── models/                 # Saved model checkpoints (excluded from Git)
├── results/                # Evaluation results (excluded from Git)
├── test_data/              # Test datasets (excluded from Git)
├── requirements.txt        # Python dependencies
├── .gitignore              # Git ignore rules
├── README.md               # This file
└── .windsurfrules          # Local environment rules (excluded from Git)
```

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/OneTrickBreach/PrepIQ---AI-NLP-Interview-Assistant.git
cd PrepIQ---AI-NLP-Interview-Assistant
```

### 2. Python Environment

- Create or activate your Conda environment or virtualenv.
- **Make sure to activate your environment before running any commands below.**
- Install dependencies:

```bash
python -m pip install -r requirements.txt
```

- Ensure you have **Google Cloud credentials** set up with access to Speech-to-Text and GCS:
    - Set `GOOGLE_APPLICATION_CREDENTIALS` environment variable to your service account JSON key file.

### 3. Frontend Setup

```bash
cd frontend
npm install
```

### 4. Google Cloud Setup

- Create a GCS bucket (done: `nlp-project-interview-audio-v2`).
- Ensure the service account has permissions to upload audio files.
- Enable Google Cloud Speech-to-Text API.

### 5. Download the Trained Model

The trained evaluation model is **not included** in this repository due to size limits.

Please download the latest trained model checkpoint from this link:

**[Your Google Drive Link Here]**

After downloading, place the `.pt` file (e.g., `custom_evaluator_best.pt`) inside the `models/` directory.

The backend will automatically load the latest model file from `models/` when started.

### 6. Generating Sample Data (Optional)

If you want to generate synthetic sample data for testing or retraining:

```bash
python scripts/generate_dataset.py
python scripts/organize_data.py
```

> **Note:** This synthetic data is very basic. For real training, replace or augment it with real, annotated examples.

### 7. Running the Backend Server

From the project root:

```bash
python -m src.main
```

### 8. Running the Frontend

In a separate terminal:

```bash
cd frontend
npm start
```

The app will be available at `http://localhost:3000`.

---

## Training the Evaluation Model

- To train from scratch:

```bash
python scripts/train_custom_evaluator.py --data_dir organized_data/ --output_dir models/ --num_epochs 20 --batch_size 8
```

- To continue training from a checkpoint:

```bash
python scripts/train_custom_evaluator.py --data_dir organized_data/ --output_dir models/ --num_epochs 10 --batch_size 8 --load_checkpoint "models/custom_evaluator_best.pt"
```

---

## Notes

- The `.gitignore` excludes large data/model directories and environment files.
- Placeholder files are included in excluded directories to preserve structure.
- You can customize roles, questions, and feedback templates by editing the data files or generation scripts.
- For speech-to-text, audio longer than 60 seconds is supported via Google Cloud Storage and asynchronous recognition.

---

## License

*Add your license information here.*

---
