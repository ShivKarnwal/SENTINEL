# SENTINEL
### AI-Based Plant Health Monitoring System

## About SENTINEL

SENTINEL is an engineering project I am developing to explore how artificial intelligence, computer vision, and embedded systems can work together to monitor plant health.
The idea is simple: instead of relying only on visual observation, I want to build a system that can analyze plant images and collect environmental information to better understand a plant's condition.
The system uses a laptop webcam to capture plant images, an Arduino Uno to collect sensor readings, and a deep learning model running locally on the laptop to classify plant diseases.
My goal is to gradually develop this into a monitoring system that can help identify possible plant health issues and track changes in plant conditions over time.

## Project Objectives

- Identify common plant diseases using image classification.
- Monitor environmental conditions around a plant.
- Collect sensor readings using an Arduino Uno.
- Analyze plant images using a deep learning model.
- Display AI predictions and sensor readings through a dashboard.
- Explore how visual and environmental information can be used together for plant monitoring.

## Hardware Components

| Component | Purpose |
|---|---|
| Laptop | Runs the AI model, dashboard, and main application |
| Laptop Webcam | Captures plant images for classification |
| Arduino Uno | Reads data from connected sensors |
| DHT22 | Measures temperature and humidity |
| LM35 | Measures temperature |
| LDR | Measures light intensity |

The laptop acts as the main processing unit. The Arduino Uno is responsible for collecting environmental sensor readings, while the laptop webcam provides images for AI-based analysis.

## Software and System Architecture

SENTINEL currently combines two main parts: image-based plant disease classification and environmental data collection.

### 1. Image Acquisition
The laptop's built-in webcam captures images of plants. These images are passed to the trained machine learning model for analysis.

### 2. Environmental Monitoring
The Arduino Uno collects readings from the connected sensors:

- DHT22 for temperature and humidity.
- LM35 for temperature.
- LDR for light intensity.

These readings provide additional information about the plant's surrounding environment.

### 3. AI-Based Plant Disease Classification
The laptop runs a trained MobileNetV3-Small model using PyTorch. The model analyzes the captured image and predicts the corresponding plant class.

### 4. Dashboard
A Streamlit-based dashboard brings the available functionality together. It provides access to webcam input, AI predictions, confidence scores, and sensor monitoring information.

### 5. Local Processing

The current implementation runs on the laptop. Image processing and AI inference do
