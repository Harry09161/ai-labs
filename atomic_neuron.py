import numpy as np


def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def atomic_neuron(a, b, c, activation=sigmoid):
    """
    Single neuron forward pass: output = activation(a * b + c)
    
    Args:
        a: input value
        b: weight
        c: bias
        activation: activation function (default: sigmoid)
    
    Returns:
        Output of the neuron
    """
    return activation(a * b + c)


def main():
    # Get user inputs
    a = float(input("Enter input value (a): "))
    b = float(input("Enter weight (b): "))
    c = float(input("Enter bias (c): "))
    l = float(input("Enter learning rate (l): "))
    y = float(input("Enter target output (y): "))
    
    # Forward pass: calculate output f
    f = atomic_neuron(a, b, c)
    
    # Calculate error e (target - predicted)
    e = y - f
     
    # Gradient descent updates (for sigmoid: derivative is f * (1 - f))
    # Weight update: b = b + l * e * f' * a, where f' = f * (1 - f) for sigmoid
    gradient = e * f * (1 - f)
    b_new = b + l * gradient * a
    c_new = c + l * gradient
    
    # Print results
    print(f"\nOutput (f): {f:.6f}")
    print(f"Error (e): {e:.6f}")
    print(f"Updated weight (b): {b_new:.6f}")
    print(f"Updated bias (c): {c_new:.6f}")
    print(f"Activation (d): sigmoid")
    print(f"Learning rate (l): {l}")
    print(f"Target (y): {y}")


if __name__ == "__main__":
    main()
