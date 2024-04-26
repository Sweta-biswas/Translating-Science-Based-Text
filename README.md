# Science Subject Translation Web App

This is a Flask web application for translating science-related text from English to Bengali using a pre-trained sequence-to-sequence model. Users can sign up, sign in, input text for translation, and download the translated text as a PDF file.

## Features

- User authentication: Users can sign up, sign in, and log out securely.
- Translation: Translate English text to Bengali using a pre-trained sequence-to-sequence model.
- PDF generation: Generate and download translated text as a PDF file.
- Input validation: Ensure that users provide valid input for translation.
- Explicit content filtering: Detect and handle explicit or bad words in the input text.

## Technologies Used

- **Flask**: Python web framework used for building the application.
- **MongoDB**: NoSQL database for storing user details and translation history.
- **PyTorch**: Deep learning library for natural language processing tasks.
- **Transformers**: Library for state-of-the-art natural language processing models.
- **IndicTrans**: Library for translating text between English and Indic languages.
- **bcrypt**: Library for securely hashing passwords.
- **pyautogui**: Library for taking screenshots.
- **img2pdf**: Library for converting images to PDF files.

## Setup Instructions

1. Clone this repository: `git clone <repository-url>`
2. Set up MongoDB and configure the connection URI in `app.py`.
3. Run the Flask application: `python app.py`
4. Access the application in your web browser at `http://localhost:5000`.

## Usage

1. Sign up for an account if you're a new user.
2. Sign in using your email and password.
3. Input English text you want to translate on the translation page.
4. Click the "Translate" button to see the translated text.
5. Optionally, click the "Download PDF" button to download the translated text as a PDF file.
6. Click the "Logout" button to sign out of your account.
