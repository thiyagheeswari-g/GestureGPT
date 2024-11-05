<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>
    <div class="container">
        <h1>GestureGPT ✋</h1>
        <p>GestureGPT is an AI-driven application designed to recognize and classify hand gestures, enabling interactions with machines through visual cues. This project leverages a Convolutional Neural Network (CNN) model to detect gestures from images, providing real-time feedback on the predicted sign.</p>
        
   <h2>🌟 Project Motivation</h2>
        <p>With the rise of AI and computer vision, gesture recognition has become an exciting field with applications in gaming, accessibility, and human-computer interaction. GestureGPT was created to explore these possibilities, aiming to develop a responsive and user-friendly model that can recognize different hand gestures with high accuracy.</p>
        
  <h2>✨ Features</h2>
  <ul>
      <li>🤖 <strong>AI-Driven Gesture Recognition</strong>: Uses a deep learning model (CNN) to classify hand gestures.</li>
      <li>📊 <strong>Performance Metrics</strong>: Provides a detailed classification report and confusion matrix to assess model accuracy.</li>
      <li>🖼️ <strong>Image Upload and Prediction</strong>: Users can upload images for real-time gesture prediction.</li>
      <li>🔍 <strong>Cross-Platform Compatibility</strong>: Designed to run seamlessly on Google Colab with Google Drive integration for easy data handling.</li>
  </ul>
        
  <h2>💻 Technologies Used</h2>
  <ul>
      <li><strong>Python</strong>: Core language used for data processing and model training.</li>
      <li><strong>TensorFlow & Keras</strong>: For building and training the Convolutional Neural Network.</li>
      <li><strong>OpenCV</strong>: Used for image processing and handling uploaded images.</li>
      <li><strong>Scikit-learn</strong>: Used for data splitting, evaluation metrics, and encoding labels.</li>
      <li><strong>Google Colab</strong>: Platform for easy access to GPUs, model training, and testing.</li>
  </ul>
  
  <h2>📸 Images</h2>
  <p>Here’s a preview of GestureGPT in action:</p>
  <img src="path/to/your/example-image.png" alt="Example Prediction">
  
  <h2>🚀 Usage</h2>
  <ol>
      <li>Clone or download the project files.</li>
      <li>Upload the project files to Google Colab.</li>
      <li>Run the cells to mount Google Drive, load the dataset, and build the CNN model.</li>
      <li>Train the model on your hand gesture dataset, then evaluate it with test data.</li>
      <li>Upload a new gesture image for real-time prediction. The system will display the predicted gesture label on the image.</li>
  </ol>
  
        
  <h2>🧠 Model Architecture Overview</h2>
  <p>The model is built using a Convolutional Neural Network (CNN) with the following layers:</p>
  <ul>
      <li><strong>Conv2D Layers:</strong> Extracts features from the input images using 32, 64, and 128 filters with ReLU activation.</li>
      <li><strong>MaxPooling2D Layers:</strong> Reduces dimensionality and computational complexity while retaining essential features.</li>
      <li><strong>Flatten Layer:</strong> Converts the matrix into a vector for fully connected layers.</li>
      <li><strong>Dense Layers:</strong> Includes a hidden layer with 128 neurons and an output layer with softmax activation to predict classes.</li>
      <li><strong>Dropout:</strong> Regularizes the model to prevent overfitting.</li>
  </ul>
  
  <h2>📊 Evaluation Metrics</h2>
  <p>The model provides a classification report and a confusion matrix to assess the model’s accuracy and performance:</p>

  <h2>📂 Project Structure</h2>
  <ul>
      <li><code>gesture_model.h5</code> - Trained CNN model for gesture recognition</li>
      <li><code>dataset/</code> - Folder containing gesture images</li>
      <li><code>GestureGPT.ipynb</code> - Main notebook for training and testing</li>
      <li><code>README.md</code> - Project documentation</li>
  </ul>
  
  <h2>🚀 Future Improvements</h2>
  <ul>
      <li>🖥️ <strong>Expand Dataset</strong>: Include more hand gestures to enhance model robustness and accuracy.</li>
      <li>📱 <strong>Mobile Compatibility</strong>: Develop a mobile version using TensorFlow Lite for on-device gesture recognition.</li>
      <li>🧑‍🤝‍🧑 <strong>Multi-User Support</strong>: Adapt model to recognize gestures from different hand shapes and skin tones.</li>
      <li>📈 <strong>Model Optimization</strong>: Experiment with advanced architectures like ResNet or MobileNet to improve accuracy and reduce latency.</li>
  </ul>
  
  <h2>🤝 Contribution</h2>
  <p>Feel free to fork the repository and submit pull requests. Suggestions for improvement are always welcome!</p>
  
  <h2>👥 Authors</h2>
  <p>Project developed by Thiyagheeswari G.</p>
  
  <div class="feature-icons">
      <span class="icon">🤖</span>
      <span class="icon">🖼️</span>
      <span class="icon">🔍</span>
      <span class="icon">📊</span>
  </div>
</div>
</body>
</html>
