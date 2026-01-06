import torch
import time
from src.engine.model import TransformerNet
from src.utils.logger import setup_logger

# Initialize professional logger
logger = setup_logger(__name__)

def test_transformer_net():
    """
    Professional test suite for TransformerNet.
    Checks:
    1. Output shape consistency.
    2. Device compatibility (CPU/CUDA).
    3. Forward pass performance (Inference speed).
    """
    logger.info("Starting TransformerNet architecture validation...")

    # 1. Device Selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Testing on device: {device}")

    try:
        # 2. Model Initialization
        model = TransformerNet().to(device)
        model.eval()  # Set to evaluation mode (important for Norm layers)

        # 3. Dummy Input Generation
        # We test with a batch size of 4 as defined in our config.yaml
        dummy_input = torch.randn(4, 3, 256, 256).to(device)
        logger.info(f"Input tensor created with shape: {dummy_input.shape}")

        # 4. Performance Measurement
        start_time = time.time()
        with torch.no_grad(): # No gradient calculation needed for testing shapes
            output = model(dummy_input)
        end_time = time.time()

        # 5. Validations
        logger.info(f"Output tensor generated with shape: {output.shape}")

        # Assertion: Check if input and output dimensions match
        assert dummy_input.shape == output.shape, \
            f"Shape Mismatch: Input {dummy_input.shape} != Output {output.shape}"

        inference_time = (end_time - start_time) * 1000 # Convert to milliseconds
        logger.info(f"✅ Forward pass successful!")
        logger.info(f"⏱️ Inference time for batch of 4: {inference_time:.2f}ms")
        logger.info("🚀 TransformerNet is ready for the training pipeline.")

    except Exception as e:
        logger.error(f"❌ Model validation failed: {str(e)}")
        raise e

if __name__ == "__main__":
    test_transformer_net()