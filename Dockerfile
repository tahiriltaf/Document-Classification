# # pack_assign/Dockerfile

# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory to /app
WORKDIR /pack_assign

# Copy the specific models into the container 
COPY models /models
# Copy the needed packages specified in requirements.txt into the container 
COPY requirements.txt ./requirements.txt
# Install any needed packages specified in requirements.txt
RUN pip3 install -r requirements.txt
COPY requirements2.txt ./requirements2.txt
RUN pip3 install -r requirements2.txt

COPY tst_images /tst_images
COPY Classify_Documents.py /pack_assign/
# Set an environment variable for the image name
ENV IMAGE_NAME=None


# Run the command to execute the Python file
CMD ["python", "Classify_Documents.py"]