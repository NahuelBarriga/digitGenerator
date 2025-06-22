import streamlit as st
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from model import CVAE, generate_digit_images  # use your functions

st.title("Handwritten Digit Generator (0–9)")
digit = st.selectbox("Select a digit to generate", list(range(10)))

if st.button("Generate Images"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CVAE()
    model.load_state_dict(torch.load("cvae_mnist.pth", map_location=device))
    model.to(device)
    images = generate_digit_images(model, digit)

    # Display images
    st.write(f"Generated images of digit: {digit}")
    fig, axs = plt.subplots(1, 5, figsize=(10, 2))
    for i in range(5):
        axs[i].imshow(images[i], cmap='gray')
        axs[i].axis('off')
    st.pyplot(fig)
