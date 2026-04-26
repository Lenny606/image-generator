from diffusers import AutoPipelineForText2Image
import torch

def main():
    print("Welcome to Image Generator!")
    print("This is a boilerplate entry point for your Python application.")

    model="stabilityai/sdxl-turbo"
    # Force float32 to avoid "Half and Float" mismatch on CPU
    pipeline = AutoPipelineForText2Image.from_pretrained(
        model, 
        torch_dtype=torch.float32,
        use_safetensors=True
    )
    pipeline.to("cpu")

    height=512
    width=512
    steps=4
    
    # 1. Consistency Configuration
    base_style = "High quality digital art, 3D Pixar style cartoon, vibrant colors, cinematic lighting"
    character_description = "A friendly golden retriever wearing a small blue collar"
    
    # 2. Story Beats (The 4 parts of your story)
    story_beats = [
        "is sitting at the front door with a happy expression, waiting for a walk",
        "is running joyfully through a field of flowers in a sunny park",
        "is jumping mid-air into a clear blue swimming pool, water splashing",
        "is curled up and sleeping soundly on a cozy rug by a fireplace at night"
    ]
    
    # Combine everything into unique prompts with a shared foundation
    prompts = [f"{base_style}. {character_description} {beat}." for beat in story_beats]
    
    # 3. Seed for consistency
    # Using the same seed helps keep the character's features similar across different prompts
    generator = torch.Generator(device="cpu").manual_seed(42)

    print(f"Generating a {len(prompts)}-part story slideshow...")
    
    # Generate all images in a batch for efficiency
    results = pipeline(
        prompt=prompts, 
        height=height, 
        width=width, 
        num_inference_steps=steps, 
        guidance_scale=0.0,
        generator=generator
    ).images

    # 4. Save with story prefix
    for i, image in enumerate(results):
        filename = f"story_{i+1}.png"
        image.save(filename)
        print(f"Saved: {filename} - Prompt: {story_beats[i]}")

if __name__ == "__main__":
    main()
