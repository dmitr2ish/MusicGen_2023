import streamlit as st
import soundfile as sf
import torch
from transformers import pipeline
import io


@st.cache_data()
def load_model():
    """ Load the AI model for music generation. """
    return pipeline("text-to-audio", "facebook/musicgen-stereo-small", device="cpu", torch_dtype=torch.float32)


# Function to generate music
def generate_music(style, description, duration):
    """ Generate music based on selected style, description, and duration. """
    synthesiser = load_model()
    prompt = f"{style} music with a {description}"
    max_new_tokens = int(duration * 25)  # Approximate conversion to tokens
    music = synthesiser(prompt, forward_params={"max_new_tokens": max_new_tokens})
    return music


# User Interface
def main():
    """ Main function to run the Streamlit app interface. """
    st.title("AI Music Generator")

    # Selection of music style
    style = st.radio(
        "Choose a music style:",
        ('Rap', 'Rock', 'Jazz', 'Electro', 'Classic', 'Blues', 'Lo-fi', 'Mumble', 'Reggie')
    )

    # Text description input
    description = st.text_input("Enter additional description:")

    # Selection of music duration
    duration = st.slider("Choose the duration (in seconds):", min_value=10, max_value=300, value=60)

    # Button to generate music
    if st.button("Generate Music"):
        with st.spinner("Generating music... This may take some time."):
            music = generate_music(style, description, duration)
            audio_file = music["audio"][0].T
            sampling_rate = music["sampling_rate"]

            # Writing audio to a byte buffer
            audio_buffer = io.BytesIO()
            sf.write(audio_buffer, audio_file, sampling_rate, format='WAV')
            audio_buffer.seek(0)

            # Display audio file and download link
            st.audio(audio_buffer, format="audio/wav")
            st.download_button(label="Download Music",
                               data=audio_buffer,
                               file_name="generated_music.wav",
                               mime="audio/wav")


if __name__ == "__main__":
    main()
