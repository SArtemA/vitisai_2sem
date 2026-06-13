# Use an official lightweight Python runtime
FROM python:3.12.13

WORKDIR /pm_2_sem

COPY requirements.txt ./

# Install dependencies without caching pip files to keep image size small
RUN pip install --no-cache-dir -q --upgrade pip && \
    pip install --no-cache-dir -q -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "./run.py"]