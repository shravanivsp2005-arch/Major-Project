# CivicPulse — AI Smart City Intelligence Platform

CivicPulse is a smart-city project designed to make it easier for residents to report local problems and for city teams to understand, prioritize, and track them. It brings civic complaints, AI-assisted classification, location-based views, and analytics into one platform.

The goal is simple: help communities and city administrators turn everyday civic issues into reports that can be routed and followed through to resolution.

## What it does

- **Civic issue reporting:** Organizes reports about roads, water and sewage, waste, traffic, electricity, public safety, and parks.
- **AI-assisted classification:** Uses text-processing and machine-learning components to help categorize complaints, suggest the responsible department, estimate priority, and identify possible duplicates.
- **Priority and SLA tracking:** Supports priority levels and target response windows so urgent issues can be surfaced sooner.
- **Complaint lifecycle:** Tracks reports through statuses such as new, assigned, in progress, resolved, reopened, or escalated.
- **City map and location context:** Provides map-oriented views to explore reports across city areas.
- **Dashboards and analytics:** Helps administrators review complaint trends, status, workload, and other reporting insights.
- **Role-aware access:** Includes authentication and protected views for different types of users.
- **Demo data support:** Includes clearly identified fictional/demo records for development and demonstrations.

> **Note:** AI predictions are decision-support suggestions. They should be reviewed by people, especially when a report involves safety or urgent public services.

## Tech stack

### Frontend
- React
- JavaScript
- React Router
- Tailwind CSS and component-based UI
- Leaflet / React Leaflet for maps
- Recharts for data visualizations

### Backend
- Python
- FastAPI
- Uvicorn
- Pydantic
- JWT-related authentication utilities

### Data, AI, and integrations
- Supabase for authentication and database access
- scikit-learn and TF-IDF-based machine-learning pipelines
- Joblib for saved model artifacts
- Ultralytics YOLO for computer-vision functionality
- Optional LLM integrations, depending on environment configuration

## Project structure

```text
Major-Project/
├── backend/
│   ├── server.py              # FastAPI application
│   ├── supabase_db.py         # Database access
│   ├── civic_supabase_auth.py # Authentication integration
│   ├── ml/                    # ML training and prediction pipeline
│   │   ├── data/              # Dataset files
│   │   └── models/             # Saved model artifacts
│   ├── vision/                # Computer-vision utilities
│   ├── tests/                 # Backend tests
│   ├── requirements.txt
│   └── .env.example
└── frontend/
    ├── src/
    │   ├── components/
    │   └── pages/
    ├── public/
    ├── package.json
    └── .env.example
```

## Getting started

### Prerequisites

Install the following before running the project:

- Python 3.10 or a compatible version supported by the dependencies
- Node.js and npm
- A Supabase project with the required database schema configured

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd Major-Project
```

If the project files are inside an additional folder in your repository, move into the directory that contains `backend/` and `frontend/`.

### 2. Configure the backend

Open a terminal in the `backend` directory:

```bash
cd backend
python -m venv .venv
```

Activate the virtual environment.

**Windows PowerShell**
```powershell
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**
```bash
source .venv/bin/activate
```

Install the Python dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Create a local environment file by copying `.env.example` to `.env`, then add your own configuration:

```env
SUPABASE_URL=your-supabase-project-url
SUPABASE_ANON_KEY=your-supabase-anon-key
SUPABASE_SERVICE_ROLE_KEY=your-server-only-service-role-key
JWT_SECRET=your-long-random-secret
CORS_ORIGINS=http://localhost:3000
```

Add optional AI-provider API keys only if you plan to use those integrations. Keep secrets in environment variables and never commit your real `.env` file or expose the Supabase service-role key in the frontend.

Set up the database using the SQL schema provided in `backend/supabase_schema.sql` and any additional setup required by your Supabase project.

Start the API:

```bash
uvicorn server:app --reload --host 127.0.0.1 --port 8000
```

The API should be available at `http://localhost:8000`. FastAPI's interactive API documentation is usually available at `http://localhost:8000/docs`.

### 3. Configure the frontend

Open another terminal:

```bash
cd frontend
npm install
```

Create a local `.env` file from `.env.example` and set the backend URL:

```env
REACT_APP_BACKEND_URL=http://localhost:8000
```

Start the frontend:

```bash
npm start
```

Open `http://localhost:3000` in your browser.

## Machine-learning models

The backend includes a machine-learning pipeline for complaint classification and prioritization. Saved model files are stored in `backend/ml/models/`, and the related training and prediction scripts are in `backend/ml/`.

If you retrain models, review the dataset paths expected by the training scripts and use data you have permission to use. Model performance depends on the quality and representativeness of the training data.

## Configuration and security

- Keep backend `.env` files and API keys private.
- Do not put the Supabase service-role key in frontend variables.
- Use a strong, unique `JWT_SECRET` outside local development.
- Configure `CORS_ORIGINS` with the exact frontend origins you trust.
- Use fictional/demo records for demonstrations unless you have permission to process real civic reports.
- Before deploying publicly, review authentication, authorization, database policies, file uploads, logging, and error handling.

## Testing

Backend tests are located in `backend/tests/`. With the backend virtual environment activated, run:

```bash
cd backend
pytest
```

The frontend includes the standard test script:

```bash
cd frontend
npm test
```

## Current status

CivicPulse is an evolving project. Setup may require adapting environment variables, database schema, and external service configuration to your own development or deployment environment. Verify each integration in your environment before relying on it in production.

## Contributing

Contributions and suggestions are welcome.

1. Fork the repository.
2. Create a branch for your change.
3. Make your changes and test them.
4. Open a pull request with a clear explanation of what changed and why.

## License

No license has been specified yet. Unless a license is added to this repository, reuse and redistribution are not automatically granted. Add a `LICENSE` file if you want to publish the project under an open-source license.

---

Built as a project exploring how AI and civic technology can support more responsive, transparent, and sustainable cities.
