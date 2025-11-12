import torch
import torch.nn as nn
import pytest

def average_gradients(models: list[torch.nn.Module]) -> None:
    """
    in-place average the gradients across all model replicas.
    """
    if len(models) == 0:
        return
    
    # Get all parameters from all models
    all_params = [list(model.parameters()) for model in models]
    
    # Average gradients for each parameter position
    for param_group in zip(*all_params):
        # Collect all gradients for this parameter
        grads = [p.grad for p in param_group if p.grad is not None]
        
        if len(grads) == 0:
            continue
            
        # Compute average
        avg_grad = torch.stack(grads).mean(dim=0)
        
        # Set averaged gradient to all models
        for p in param_group:
            if p.grad is not None:
                p.grad.copy_(avg_grad)


# Unit Tests
def test_gradient_averaging_correctness():
    """Test that gradients are correctly averaged"""
    # Create two models with same architecture
    model1 = nn.Linear(4, 2, bias=False)
    model2 = nn.Linear(4, 2, bias=False)
    
    # Set known gradients manually
    model1.weight.grad = torch.tensor([[1.0, 2.0, 3.0, 4.0],
                                        [5.0, 6.0, 7.0, 8.0]])
    model2.weight.grad = torch.tensor([[3.0, 4.0, 5.0, 6.0],
                                        [7.0, 8.0, 9.0, 10.0]])
    
    expected_avg = torch.tensor([[2.0, 3.0, 4.0, 5.0],
                                  [6.0, 7.0, 8.0, 9.0]])
    
    average_gradients([model1, model2])
    
    assert torch.allclose(model1.weight.grad, expected_avg)
    assert torch.allclose(model2.weight.grad, expected_avg)
    print("✓ Gradient averaging correctness test passed")


def test_all_models_identical_gradients():
    """Test that all models end up with identical gradients"""
    models = [nn.Linear(4, 2, bias=False) for _ in range(4)]
    
    # Set different gradients for each
    for i, model in enumerate(models):
        model.weight.grad = torch.randn(2, 4) * (i + 1)
    
    average_gradients(models)
    
    # All gradients should now be identical
    reference_grad = models[0].weight.grad
    for model in models[1:]:
        assert torch.allclose(model.weight.grad, reference_grad)
    print("✓ All models identical gradients test passed")


def test_single_model():
    """Test with a single model (edge case)"""
    model = nn.Linear(4, 2, bias=False)
    original_grad = torch.randn(2, 4)
    model.weight.grad = original_grad.clone()
    
    average_gradients([model])
    
    # Gradient should remain unchanged
    assert torch.allclose(model.weight.grad, original_grad)
    print("✓ Single model test passed")


def test_three_models():
    """Test averaging with three models"""
    models = [nn.Linear(3, 2, bias=False) for _ in range(3)]
    
    models[0].weight.grad = torch.ones(2, 3) * 3.0
    models[1].weight.grad = torch.ones(2, 3) * 6.0
    models[2].weight.grad = torch.ones(2, 3) * 9.0
    
    expected = torch.ones(2, 3) * 6.0  # (3 + 6 + 9) / 3 = 6
    
    average_gradients(models)
    
    for model in models:
        assert torch.allclose(model.weight.grad, expected)
    print("✓ Three models test passed")


def test_multi_layer_model():
    """Test with models having multiple parameters"""
    class MultiLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(4, 3, bias=False)
            self.fc2 = nn.Linear(3, 2, bias=False)
    
    models = [MultiLayer(), MultiLayer()]
    
    # Set gradients for both layers
    models[0].fc1.weight.grad = torch.ones(3, 4) * 2.0
    models[0].fc2.weight.grad = torch.ones(2, 3) * 4.0
    models[1].fc1.weight.grad = torch.ones(3, 4) * 6.0
    models[1].fc2.weight.grad = torch.ones(2, 3) * 8.0
    
    average_gradients(models)
    
    # Check both layers
    assert torch.allclose(models[0].fc1.weight.grad, torch.ones(3, 4) * 4.0)
    assert torch.allclose(models[1].fc1.weight.grad, torch.ones(3, 4) * 4.0)
    assert torch.allclose(models[0].fc2.weight.grad, torch.ones(2, 3) * 6.0)
    assert torch.allclose(models[1].fc2.weight.grad, torch.ones(2, 3) * 6.0)
    print("✓ Multi-layer model test passed")


def test_realistic_training_scenario():
    """Test with actual forward/backward pass"""
    torch.manual_seed(42)
    models = [nn.Linear(4, 2, bias=False) for _ in range(2)]
    
    # Initialize with same weights (simulating replicated model)
    with torch.no_grad():
        for model in models[1:]:
            model.weight.copy_(models[0].weight)
    
    loss_fn = nn.MSELoss()
    x1, y1 = torch.randn(5, 4), torch.randn(5, 2)
    x2, y2 = torch.randn(5, 4), torch.randn(5, 2)
    
    # Different data on each replica
    for m, (x, y) in zip(models, [(x1, y1), (x2, y2)]):
        out = m(x)
        loss = loss_fn(out, y)
        loss.backward()
    
    # Store original gradients
    grad1 = models[0].weight.grad.clone()
    grad2 = models[1].weight.grad.clone()
    
    # Gradients should be different before averaging
    assert not torch.allclose(grad1, grad2)
    
    average_gradients(models)
    
    # After averaging, should be identical and equal to manual average
    expected = (grad1 + grad2) / 2
    assert torch.allclose(models[0].weight.grad, expected, atol=1e-6)
    assert torch.allclose(models[1].weight.grad, expected, atol=1e-6)
    print("✓ Realistic training scenario test passed")


if __name__ == "__main__":
    test_gradient_averaging_correctness()
    test_all_models_identical_gradients()
    test_single_model()
    test_three_models()
    test_multi_layer_model()
    test_realistic_training_scenario()
    print("\n✅ All tests passed!")