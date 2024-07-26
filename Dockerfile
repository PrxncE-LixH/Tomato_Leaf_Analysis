FROM python

#CREATE A DIRECTORY APP and copy requirements.txt into it
COPY requirements.txt /app/

#COPY everything into the app directory
COPY . /app/ 


#CHANGE DIRECTORY to the app
WORKDIR /app/


# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt


# Run the application
CMD ["uvicorn", "backend:app", "--host=0.0.0.0", "--port=80"]
