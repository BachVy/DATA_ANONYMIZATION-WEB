# DATA_ANONYMIZATION-WEB
A web application for data anonymization using the k-anonymity principle and the APO_ARO (Artificial Predator Optimization - Artificial Rabbit Optimization) algorithm.
## List of Contents
- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Notes](#notes)
- [License](#license)

## Overview
`DATA_ANONYMIZATION-WEB` is a Flask-based web application that allows users to anonymize datasets while preserving privacy using k-anonymity and the APO_ARO algorithm. The application supports user authentication, guest mode, theme switching (light/dark), and provides a user-friendly interface to upload datasets, configure anonymization parameters, and download results.

## Features
- **User Authentication**: Register, log in, or use guest mode to access the application.
- **Data Anonymization**: Upload CSV files and anonymize them using k-anonymity and APO_ARO.
- **Theme Switching**: Switch between light and dark themes.
- **History Tracking**: View anonymization history (for logged-in users).
- **Preview and Download**: Preview anonymized data and download the results.
- **Policy Agreement**: Users must agree to the usage policy before accessing the application.

## Requirements
- Python 3.8 or higher
- Git (for cloning the repository)
- Web browser (e.g., Chrome, Firefox)

## Installation
1. **Clone the Repository**:
   ```
   git clone https://github.com/your-username/DATA_ANONYMIZATION-WEB.git
   cd DATA_ANONYMIZATION-WEB
2. **Set Up a Virtual Environment (optional but recommended):**
```
python -m venv venv
source venv/bin/activate  # On Linux/macOS
venv\Scripts\activate     # On Windows
```
3. **Install Dependencies:**
```
pip install -r requirements.txt
```
4. **Prepare Data:**
Place your input CSV file in the data/ directory (e.g., data/your_dataset.csv).
Ensure the hierarchies/ directory contains hierarchy files for each column (e.g., hierarchy_age.csv).
5. **Run the Application:**
```
python app.py
```
The app will be available at http://127.0.0.1:5000.
## Usage
1. **Access the Application:**
Open your browser and navigate to http://127.0.0.1:5000.
2. **Register or Log In:**
Go to /register to create a new account, or /login to log in.
Alternatively, use "Guest Login" to access the app without an account.
Agree to the usage policy at /policy to proceed.
3. **Anonymize Data:**
- Navigate to /anonymize.
- Upload a CSV file (must use ; as the delimiter).
- Enter the k value for k-anonymity (e.g., 5).
- (Optional) Specify a target column for classification accuracy calculation.
- Click "Ẩn danh" to anonymize the data.
- View the preview of the anonymized data and metrics (Information Loss, Classification Accuracy, etc.).
- Download the anonymized file or reset the page to upload a new file.
4. **Manage Profile (Logged-in Users):**
Go to /user_profile to change the theme (light/dark) or view/delete anonymization history.
5. **Log Out:**
Click "Đăng xuất" at /logout to end your session.
### Project Structure
```
DATA_ANONYMIZATION-WEB/
├── data/                          # Input data and user data
│   └── users.csv                  # User data (auto-generated)
│
├── hierarchies/                   # Hierarchy files for anonymization
│   ├── history.csv                # Example hierarchy file
│   └── users.csv                  # Example hierarchy file
│
├── static/                        # Static files (CSS, JS, images)
│   ├── background.jpg             # Background image
│   ├── logo1.png                  # Logo image
│   ├── script.js                  # JavaScript for frontend logic
│   └── style.css                  # CSS styles
│
├── templates/                     # HTML templates
│   ├── anonymize.html             # Anonymization page
│   ├── index.html                 # Home page
│   ├── login.html                 # Login page
│   ├── policy.html                # Policy page
│   ├── register.html              # Registration page
│   └── user_profile.html          # User profile page
│
├── uploads/                       # Temporary storage for uploaded files
│
├── results/                       # Storage for anonymized files and train/val indexes
│
├── anonymize.py                   # Functions for data anonymization
├── apo_aro.py                     # APO_ARO algorithm implementation
├── app.py                         # Main Flask application
├── fitness.py                     # Fitness function for optimization
├── split.py                       # Data splitting for train/test
├── train_dynamic.py               # Core anonymization logic
├── requirements.txt               # List of dependencies
└── README.md                      # Project documentation
```
## How It Works
### User Authentication:
- User data is stored in data/users.csv with hashed passwords (using bcrypt).
- Session management is handled by Flask.
### Data Anonymization:
- The app uses k-anonymity to ensure privacy, with the APO_ARO algorithm to optimize generalization levels.
- Input CSV files are processed, and hierarchies are applied to generalize data.
- Metrics like Information Loss (IL), Classification Accuracy (CA), and - - - Penalty (m) are calculated.
### Frontend:
- Built with HTML, CSS, and JavaScript.
- Supports light/dark themes and a loading spinner during anonymization.
### Input File Format:
- CSV files must use ; as the delimiter.
- Ensure hierarchy files in hierarchies/ match the columns in your dataset.
### Performance:
Large datasets or high k values may increase processing time.
### Security:
- Passwords are hashed using bcrypt.
- Set a secure app.secret_key in app.py to protect sessions.
### Temporary Files:
Uploaded and anonymized files are stored in uploads/ and results/. These directories are ignored by Git (via .gitignore).