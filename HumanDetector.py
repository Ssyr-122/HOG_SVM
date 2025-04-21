import cv2
import numpy as np
import os
import glob
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import time

class HOGSVMPipeline:
    def __init__(self, hog_params=None):
        # Default HOG parameters
        if hog_params is None:
            self.hog_params = {
                'winSize': (64, 128),          # Standard size for human detection
                'blockSize': (16, 16),         # Block size
                'blockStride': (8, 8),         # Block stride
                'cellSize': (8, 8),            # Cell size
                'nbins': 9,                    # Number of bins
                'derivAperture': 1,            # Sobel kernel size
                'winSigma': -1,                # Gaussian smoothing window parameter
                'histogramNormType': 0,        # L2-Hys norm
                'L2HysThreshold': 0.2,         # L2-Hys normalization method shrinkage
                'gammaCorrection': False,       # Whether to do gamma correction
                'nlevels': 64,                 # Number of detection window size levels
                'signedGradient': False        # Whether to use signed gradient
            }
        else:
            self.hog_params = hog_params
            
        # Initialize HOG descriptor
        self.hog = cv2.HOGDescriptor(
            self.hog_params['winSize'],
            self.hog_params['blockSize'],
            self.hog_params['blockStride'],
            self.hog_params['cellSize'],
            self.hog_params['nbins'],
            self.hog_params['derivAperture'],
            self.hog_params['winSigma'],
            self.hog_params['histogramNormType'],
            self.hog_params['L2HysThreshold'],
            self.hog_params['gammaCorrection'],
            self.hog_params['nlevels'],
            self.hog_params['signedGradient']
        )
        
        # SVM classifier
        self.classifier = svm.SVC(kernel='linear', probability=True)
        
    def preprocess_image(self, image_path):
        """Load and preprocess an image to the correct size for HOG."""
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                print(f"Warning: Could not read image {image_path}")
                return None
                
            # Resize to HOG window size
            win_size = self.hog_params['winSize']
            img = cv2.resize(img, win_size)
            
            # Convert to grayscale if needed
            if len(img.shape) > 2 and img.shape[2] > 1:
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                img_gray = img
                
            return img_gray
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None
    
    def extract_features(self, image_path):
        """Extract HOG features from a single image."""
        img = self.preprocess_image(image_path)
        if img is None:
            return None
            
        # Compute HOG features
        features = self.hog.compute(img)
        return features
    
    def load_dataset(self, human_dir, non_human_dir):
        features = []
        labels = []
        image_paths = []

        # Load human images
        human_images = glob.glob(os.path.join(human_dir, "*.*"))
        for img_path in human_images:
            feat = self.extract_features(img_path)
            if feat is not None:
                features.append(feat.flatten())
                labels.append(1)
                image_paths.append(img_path)

        # Load non-human images
        non_human_images = glob.glob(os.path.join(non_human_dir, "*.*"))
        for img_path in non_human_images:
            feat = self.extract_features(img_path)
            if feat is not None:
                features.append(feat.flatten())
                labels.append(0)
                image_paths.append(img_path)

        return np.array(features), np.array(labels), image_paths
    
    def train(self, X_train, y_train):
        """Train the SVM classifier."""
        start_time = time.time()
        self.classifier.fit(X_train, y_train)
        train_time = time.time() - start_time
        print(f"Training completed in {train_time:.2f} seconds")
    
    def evaluate(self, X_test, y_test, image_paths=None):
        start_time = time.time()
        y_pred = self.classifier.predict(X_test)
        test_time = time.time() - start_time

        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=['Non-Human', 'Human'])

        print(f"Testing completed in {test_time:.2f} seconds")
        print(f"Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(report)

        # Show misclassified images
        if image_paths:
            print("\nMisclassified Images:")
            for i, (pred, true) in enumerate(zip(y_pred, y_test)):
                if pred != true:
                    print(f"- {image_paths[i]}: predicted {pred}, actual {true}")

        return accuracy, report, y_pred
    
    def save_model(self, model_path):
        """Save the trained model."""
        with open(model_path, 'wb') as f:
            pickle.dump(self.classifier, f)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path):
        """Load a trained model."""
        with open(model_path, 'rb') as f:
            self.classifier = pickle.load(f)
        print(f"Model loaded from {model_path}")
    
    def predict(self, image_path):
        """Predict whether an image contains a human."""
        features = self.extract_features(image_path)
        if features is None:
            return None
        
        prediction = self.classifier.predict([features.flatten()])[0]
        probability = self.classifier.predict_proba([features.flatten()])[0]
        
        result = {
            'is_human': bool(prediction),
            'confidence': probability[1] if prediction == 1 else probability[0]
        }
        
        return result
    
    def visualize_results(self, X_test, y_test, save_dir="."):
        """Create and save visualizations for model evaluation."""
        # Ensure save directory exists
        os.makedirs(save_dir, exist_ok=True)
        
        # Get predictions and probabilities
        y_pred = self.classifier.predict(X_test)
        y_prob = self.classifier.predict_proba(X_test)[:, 1]  # Probability for the positive class (Human)
        
        # 1. Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Non-Human', 'Human'], 
                    yticklabels=['Non-Human', 'Human'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300)
        plt.close()


# Example usage with your dataset specifications
if __name__ == "__main__":
    # Directories
    train_human_dir = "dataset/human_train"
    train_non_human_dir = "dataset/non-human_train"
    test_human_dir = "dataset/human_test"
    test_non_human_dir = "dataset/non-human_test"

    print("program starting...")
    
    # Initialize pipeline
    pipeline = HOGSVMPipeline()
    
    # Training phase
    print("Loading training data...")
    X_train, y_train, train_paths = pipeline.load_dataset(train_human_dir, train_non_human_dir)
    print(f"Loaded {len(X_train)} training samples ({sum(y_train)} humans, {len(y_train) - sum(y_train)} non-humans)")
    
    print("Training SVM classifier...")
    pipeline.train(X_train, y_train)
    
    # Save model (optional)
    pipeline.save_model("hog_svm_model.pkl")
    
    # Testing phase
    print("Loading test data...")
    X_test, y_test, test_image_paths = pipeline.load_dataset(test_human_dir, test_non_human_dir)
    print(f"Loaded {len(X_test)} test samples ({sum(y_test)} humans, {len(y_test) - sum(y_test)} non-humans)")

    print("Evaluating model...")
    accuracy, report, y_pred = pipeline.evaluate(X_test, y_test, image_paths=test_image_paths)
    
    # Example prediction on a single image
    test_image = "path/to/test/image.jpg"
    if os.path.exists(test_image):
        result = pipeline.predict(test_image)
        if result:
            print(f"Prediction for {test_image}: {'Human' if result['is_human'] else 'Non-Human'} with {result['confidence']:.2f} confidence")

    # Create a results directory
    results_dir = "visualization_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # After evaluation, add:
    print("Generating visualizations...")
    pipeline.visualize_results(X_test, y_test, save_dir=results_dir)