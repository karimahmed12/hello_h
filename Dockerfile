# Use the official Python image.
# https://hub.docker.com/_/python
FROM python:3.10-slim

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install nltk
RUN python -c "import nltk; nltk.download('wordnet')"
# Download NLTK data
RUN python -m nltk.downloader stopwords

RUN python -m nltk.downloader punkt wordnet

RUN python -m nltk.downloader punkt
# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["python", "app.py"]
