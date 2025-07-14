# Taxi Fare Prediction

This is a simple machine learning web service built using Flask that predicts the taxi fare in New York City based on input features like trip distance, passenger count, and hour of the day.

## Features

- Linear Regression model trained on NYC Taxi Fare data
- Flask API for real-time predictions
- Hosted on AWS EC2
- Accepts JSON POST requests

Project Structure
bash
Copy
Edit
taxi-fare-app/
├── app.py              # Flask API server
├── fare_model.pkl      # Trained ML model
├── train.py            # Script to train the model
├── taxi.csv            # Dataset (optional, if committed)
├── venv/               # Python virtual environment

How to Run
bash
Copy
Edit
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py

## Example Request

```bash
curl -X POST http://<your-ec2-public-ip>:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"trip_distance": 2.3, "passenger_count": 1, "hour": 14}'

Response
json
Copy
Edit
{
  "fare": 6.37
}



