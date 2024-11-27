# Specify base ikmage for docker container
FROM python:3.14-rc-bookworm
# Sets working directory inside docker container
WORKDIR /zee-recommender

# Below lines ensures that pip is upgraded to latest version
RUN python -m pip install --upgrade pip
# Copy requirements/dependencies from local machine to container
COPY requirements.txt requirements.txt
# Run requirements.txt recursively to install all required libraries
RUN pip install -r requirements.txt

# Copy entire project folder from local machine to container
COPY . .

# Define entry point for Flask application
ENV FLASK_APP=main.py

# Specify default command to run when container starts
CMD ["python","-m","flask","run","--host=0.0.0.0"]

# To build docker container > docker build -t <tag_name>
# To run docker container > docker run -p <docker_port>:<local_machine_port(5000 for flask)> -d <tag_name>
