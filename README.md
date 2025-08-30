# Emotion Detection Web Application

A full-stack web application for detecting emotions in images using deep learning. The application allows users to upload images or capture them using their webcam, and then uses a pre-trained model to detect the dominant emotion in the image.

## Features

- **Image Upload**: Upload images from your device
- **Webcam Support**: Capture images directly from your webcam
- **Real-time Analysis**: Get instant emotion detection results
- **Responsive Design**: Works on desktop and mobile devices
- **Contact Form**: Send messages to the support team

## Tech Stack

- **Frontend**: React.js, Vite, Tailwind CSS, Axios
- **Backend**: FastAPI, Python 3.9+
- **ML Model**: Pre-trained Keras model (TensorFlow)
- **Containerization**: Docker, Docker Compose
- **Web Server**: Nginx (production), Uvicorn (development)

## Prerequisites

- Docker and Docker Compose installed on your system
- Node.js (for local frontend development)
- Python 3.9+ (for local backend development)

## Getting Started

### Using Docker (Recommended)

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd emotion-detection-app
   ```

2. Copy your pre-trained model file to `backend/model/emotion_model.h5`

3. Start the application using Docker Compose:
   ```bash
   docker-compose up --build
   ```

4. The application will be available at:
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

### Local Development

#### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Start the backend server:
   ```bash
   uvicorn app.main:app --reload
   ```

#### Frontend Setup

1. In a new terminal, navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```

4. The frontend will be available at http://localhost:5173

## Project Structure

```
emotion-detection-app/
├── backend/                 # Backend (FastAPI)
│   ├── app/                 # Application code
│   │   └── main.py          # Main FastAPI application
│   ├── model/               # Model directory
│   │   └── emotion_model.h5 # Pre-trained model (not included in repo)
│   ├── Dockerfile           # Backend Dockerfile
│   └── requirements.txt     # Python dependencies
│
├── frontend/                # Frontend (React)
│   ├── public/              # Static files
│   ├── src/                 # React source code
│   │   ├── components/      # Reusable components
│   │   ├── pages/           # Page components
│   │   ├── App.jsx          # Main App component
│   │   └── main.jsx         # Entry point
│   ├── Dockerfile           # Frontend Dockerfile
│   ├── nginx.conf           # Nginx configuration
│   └── package.json         # Frontend dependencies
│
├── docker-compose.yml       # Docker Compose configuration
└── README.md               # This file
```

## Environment Variables

### Backend

Create a `.env` file in the `backend` directory with the following variables:

```
# FastAPI
APP_NAME=Emotion Detection API
APP_VERSION=1.0.0
DEBUG=True

# CORS
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5173

# Email (for contact form)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=your-password
EMAIL_FROM=your-email@gmail.com
```

## API Endpoints

- `GET /api/health` - Health check endpoint
- `POST /api/predict` - Predict emotion from an image
  - Method: POST
  - Content-Type: multipart/form-data
  - Body: `file` (image file)
  - Response: `{ "emotion": "Happy", "confidence": 0.92 }`
- `POST /api/contact` - Submit contact form
  - Method: POST
  - Content-Type: application/json
  - Body: `{ "name": "John Doe", "email": "john@example.com", "message": "Hello!" }`

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/)
- [React](https://reactjs.org/)
- [TensorFlow](https://www.tensorflow.org/)
- [Tailwind CSS](https://tailwindcss.com/)
