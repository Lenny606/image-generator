from diffusers import AutoPipelineForText2Image
import torch

def main():
    print("Welcome to Image Generator!")
    print("This is a boilerplate entry point for your Python application.")

    model="stabilityai/sdxl-turbo"
    pipeline = AutoPipelineForText2Image.from_pretrained(model).to('cpu')

    height=512
    width=512
    steps=1

    prompt="a photo of dog, black and white, looking at camera"

    image= pipeline(prompt=prompt, height=height, width=width, num_inference_steps=steps, guidance_scale=0.0).images[0]

    image.save("dog.png")
    

if __name__ == "__main__":
    main()
