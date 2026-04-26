# Project Overview
`image-generator` is a Python-based application designed to generate sequential image stories using Stable Diffusion. It leverages the `diffusers` library with the `stabilityai/sdxl-turbo` model to create high-quality, stylistically consistent images from text prompts.

The project is structured as a simple script-based application, currently focused on a fixed narrative featuring a golden retriever, but easily adaptable for other subjects and styles.

## Tech Stack
- **Language:** Python 3.12+
- **Dependency Management:** [Poetry](https://python-poetry.org/)
- **Image Generation:** [Diffusers](https://huggingface.co/docs/diffusers/index) (Stable Diffusion XL Turbo)
- **Machine Learning Backend:** [PyTorch](https://pytorch.org/) (configured for CPU usage by default)
- **Image Processing:** [Pillow (PIL)](https://python-pillow.org/)

## Building and Running
The project uses Poetry for environment management.

- **Install Dependencies:**
  ```bash
  poetry install
  ```
- **Run the Application:**
  ```bash
  poetry run python main.py
  ```
- **Run Tests (when implemented):**
  ```bash
  poetry run pytest
  ```

## Development Conventions
- **Model Selection:** Currently uses `stabilityai/sdxl-turbo` for fast inference.
- **Hardware Compatibility:** Defaulted to `torch.float32` and `device="cpu"` to ensure compatibility across environments without requiring a dedicated GPU.
- **Consistency:** Employs a fixed seed (`42`) and a shared `base_style` prompt prefix to maintain visual coherence across generated image sequences.
- **Outputs:** Generated images are saved in the root directory with the naming convention `story_N.png`.
