# Random Client Selection (FedAvg) Federated Learning on MNIST

This project implements a federated learning simulation using the Flower (flwr) framework with the FedAvg strategy and random client selection. It trains a simple neural network on the MNIST dataset to classify handwritten digits, distributed across 10 clients.

## Table of Contents
- [Project Overview](#project-overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
This project demonstrates federated learning using the Flower framework. Key features include:
- **Federated Learning Setup**: Simulates 10 clients, each with a subset of the MNIST dataset.
- **Strategy**: Uses FedAvg with random client selection (50% of clients per round).
- **Model**: A simple neural network with two dense layers for digit classification.
- **Metrics**: Tracks communication rounds, total data transferred, final accuracy, and convergence speed.

The code is designed to run in environments like Google Colab or locally, with minimal setup.

## Requirements
- Python 3.8+
- Flower (`flwr[simulation]`)
- TensorFlow
- NumPy

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/federated-learning-flower-mnist.git
   cd federated-learning-flower-mnist
   ```
2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install the required packages:
   ```bash
   pip install -U "flwr[simulation]" tensorflow numpy
   ```

## Usage
1. Ensure the dependencies are installed as described above.
2. Run the script:
   ```bash
   python federated_learning_flower.py
   ```
3. The script will:
   - Load and split the MNIST dataset across 10 clients.
   - Simulate federated learning for 10 communication rounds.
   - Print metrics including communication rounds, total data transferred, final accuracy, and convergence speed.

### Running in Google Colab
1. Open Google Colab and create a new notebook.
2. Install dependencies in a cell:
   ```bash
   !pip install -U "flwr[simulation]" tensorflow numpy
   ```
3. Copy the code from `federated_learning_flower.py` into a new cell and run it.
4. Adjust `NUM_CLIENTS` or `NUM_ROUNDS` if you encounter memory issues.

## Dataset
The project uses the **MNIST dataset**, a standard benchmark for digit classification:
- **Content**: 70,000 grayscale images of handwritten digits (0-9).
- **Split**: 60,000 training images, 10,000 test images.
- **Image Size**: 28x28 pixels (784 pixels total).
- **Preprocessing**: Pixel values are normalized to [0, 1].
- **Federated Setup**: The training data is evenly split across 10 clients (6,000 samples per client).

## Results
Sample output from a run with 10 clients and 10 rounds:
```
Number of communication rounds: 10
Total data transferred: 38.82 MB
Final model accuracy: 0.9651
Convergence speed (avg accuracy improvement per round): 0.006641
```
- **Total Data Transferred**: 38.82 MB over 10 rounds (approximately 3.88 MB per round for 5 active clients).
- **Accuracy**: Achieved 96.51% on the test set after 10 rounds.
- **Convergence Speed**: Average accuracy improvement of 0.006641 per round.

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Make your changes and commit (`git commit -m "Add your feature"`).
4. Push to your branch (`git push origin feature/your-feature`).
5. Open a pull request.

Please ensure your code follows the existing style and includes appropriate comments.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.