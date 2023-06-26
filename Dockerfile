# # Use the official PyTorch image as a base
# FROM pytorch/pytorch:latest

# # Set the working directory
# WORKDIR /app

# # Copy the script into the container
# COPY main.py /app/

# # When the container is run, execute the script
# CMD ["python", "/app/main.py"]

FROM nvidia/cuda:12.0.0-devel-ubuntu20.04

WORKDIR /usr/src/app

COPY matrix_mul.cu .

RUN nvcc -o matrix_mul matrix_mul.cu

CMD ["./matrix_mul"]
