# Trained on the following Tomato leaves
- Bacterial Spot, Target Spot, Mosaic Virus, YellowLeaf Curl Virus, Healthy

## BACKEND
# pip install requirements.txt to install dependencies needed for the backend
- uvicorn backend:app --reload   to run the backend

## FRONTEND
# To run, npm install package-lock.json in frontend
- make these changes  in your scripts section in the package.json file to get it to work 
- "start": "react-scripts --openssl-legacy-provider start",
- "build": "react-scripts --openssl-legacy-provider build",
- create a .env file in the root directory of the frontend folder with the backend url -  REACT_APP_API_URL="http://localhost:8000/predict"
- npm run start in the frontend folder to run the webpage