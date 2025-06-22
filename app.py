# app.py

import streamlit as st
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- CVAE Model Definition ---
class CVAE(nn.Module):
    def __init__(self, latent_dim=20):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.fc1 = nn.Linear(784 + 10, 400)
        self.fc21 = nn.Linear(400, latent_dim)
        self.fc22 = nn.Linear(400, latent_dim)
        self.fc3 = nn.Linear(latent_dim + 10, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x, y):
        h1 = F.relu(self.fc1(torch.cat([x, y], dim=1)))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y):
        h3 = F.relu(self.fc3(torch.cat([z, y], dim=1)))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, y), mu, logvar

# --- Load Trained Model ---
@st.cache_resource
def load_model(model_path="cvae_mnist.pth"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CVAE()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device

# --- Generate Images ---
def generate_digit_images(model, digit, num_images=5, device='cpu'):
    y = torch.eye(10)[[digit] * num_images].to(device)
    z = torch.randn(num_images, model.latent_dim).to(device)
    with torch.no_grad():
        generated = model.decode(z, y).cpu().view(-1, 28, 28)
    return generated

# --- Streamlit Interface ---
st.set_page_config(page_title="Digit Generator", layout="centered")
st.title("🧠 Handwritten Digit Generator (0–9)")
st.write("Select a digit and generate 5 handwritten samples using a CVAE model trained on MNIST.")

digit = st.selectbox("Choose a digit to generate", list(range(10)), index=0)

if st.button("🚀 Generate Images"):
    st.info(f"Generating 5 images of digit `{digit}`...")
    model, device = load_model()
    images = generate_digit_images(model, digit, device=device)

    fig, axs = plt.subplots(1, 5, figsize=(10, 2))
    for i in range(5):
        axs[i].imshow(images[i], cmap="gray")
        axs[i].axis("off")
    st.pyplot(fig)
