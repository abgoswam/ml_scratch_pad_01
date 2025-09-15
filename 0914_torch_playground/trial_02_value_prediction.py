import torch
import torch.nn as nn

class ValuePredictionNetwork(nn.Module):
    """
    Value Prediction Network that takes transformer output and predicts a scalar value
    representing expected reward from the current position.
    """
    
    def __init__(self, input_dim=768):
        super().__init__()
        self.input_dim = input_dim
        # Linear layer: input_dim -> 1 (includes weight matrix W and bias b)
        self.value_head = nn.Linear(input_dim, 1)
    
    def forward(self, transformer_output):
        """
        Forward pass through value network
        
        Args:
            transformer_output: [batch_size, 768] or [768] - output from transformer
            
        Returns:
            value: scalar or [batch_size] - predicted value(s)
        """
        # Step 1: input = transformer_output [768] or [batch_size, 768]
        input_tensor = transformer_output
        
        # Step 2: weighted_sum = W @ input + b [1] or [batch_size, 1]
        weighted_sum = self.value_head(input_tensor)
        
        # Step 3: value = squeeze to remove dimension of size 1
        value = weighted_sum.squeeze(-1)  # Remove last dimension if size 1
        
        return value

# Example usage
def demo_value_prediction():
    print("=== Value Prediction Network Demo ===\n")
    
    # Create the network
    value_net = ValuePredictionNetwork(input_dim=768)
    
    # Example 1: Batch transformer output with shape (2, 768)
    print("Example 1: Batch transformer output (2, 768)")
    transformer_output = torch.tensor([
        [0.2, 0.6, -0.2, 0.9, 0.1, -0.3, 0.4, -0.1] + [0.0] * (768 - 8),
        [0.5, -0.1, 0.3, -0.7, 0.8, 0.2, -0.4, 0.6] + [0.1] * (768 - 8)
    ])  # Shape: (2, 768)
    
    print(f"Transformer output shape: {transformer_output.shape}")
    
    # Manual computation steps (as you outlined)
    print("\nManual computation steps:")
    print("1. input = transformer_output  # [2, 768]")
    
    # Get weight matrix and bias from the linear layer
    W = value_net.value_head.weight  # [1, 768]
    b = value_net.value_head.bias    # [1]
    
    print(f"2. W shape: {W.shape}, b shape: {b.shape}")
    
    # Matrix multiplication for batch
    weighted_sum = torch.matmul(transformer_output, W.T)  # [2, 1]
    print(f"3. weighted_sum = input @ W.T: {weighted_sum.shape}")
    
    # Add bias
    value_with_bias = weighted_sum + b  # [2, 1]
    print(f"4. value = weighted_sum + bias: {value_with_bias.shape}")
    
    # Squeeze to remove last dimension
    final_value = value_with_bias.squeeze(-1)  # [2]
    print(f"5. return value.squeeze(-1): {final_value.shape} -> {final_value}")
    
    # Using the network directly
    print("\nUsing ValuePredictionNetwork:")
    predicted_value = value_net(transformer_output)
    print(f"Predicted values shape: {predicted_value.shape}")
    print(f"Predicted values: {predicted_value}")
    
    # Example 2: Batch of transformer outputs
    print("\n" + "="*50)
    print("Example 2: Larger batch processing")
    
    batch_size = 4
    batch_transformer_output = torch.randn(batch_size, 768)
    print(f"Batch transformer output shape: {batch_transformer_output.shape}")
    
    batch_values = value_net(batch_transformer_output)
    print(f"Batch predicted values shape: {batch_values.shape}")
    print(f"Batch predicted values: {batch_values}")
    
    # Example 3: Show the squeeze operation in detail
    print("\n" + "="*50)
    print("Example 3: Demonstrating squeeze operation")
    
    # Before squeeze
    raw_output = value_net.value_head(transformer_output)
    print(f"Raw linear layer output shape: {raw_output.shape}")  # [2, 1]
    print(f"Raw linear layer output: {raw_output}")
    
    # After squeeze
    squeezed_output = raw_output.squeeze(-1)
    print(f"After squeeze shape: {squeezed_output.shape}")  # [2]
    print(f"After squeeze: {squeezed_output}")

# Example 4: Compare with and without squeeze
def compare_squeeze_behavior():
    print("\n" + "="*50)
    print("Example 4: With vs Without Squeeze")
    
    value_net = ValuePredictionNetwork(input_dim=768)
    transformer_output = torch.randn(2, 768)  # Changed to (2, 768)
    
    # Without squeeze (keeping dimension)
    raw_value = value_net.value_head(transformer_output)
    print(f"Without squeeze - shape: {raw_value.shape}, value: {raw_value.flatten()}")
    
    # With squeeze (removing dimension)
    squeezed_value = raw_value.squeeze(-1)
    print(f"With squeeze - shape: {squeezed_value.shape}, value: {squeezed_value}")
    
    # Batch example
    batch_output = torch.randn(3, 768)
    batch_raw = value_net.value_head(batch_output)
    batch_squeezed = batch_raw.squeeze(-1)
    
    print(f"Batch without squeeze - shape: {batch_raw.shape}")
    print(f"Batch with squeeze - shape: {batch_squeezed.shape}")

if __name__ == "__main__":
    demo_value_prediction()
    compare_squeeze_behavior()