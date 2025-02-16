import time
import os
import streamlit as st
from huggingface_hub import InferenceClient
from PIL import Image
from io import BytesIO



# Streamlit UI
st.title("Choose Your Customizable Event Invitation Cards")
st.write("Generate a colorful invitation card for your event with a theme of your choice.")

# User input for customization
color = st.selectbox("Pick a color:",["red","blue","green","brown","purple","yellow","pink","black","white","orange"])
event = st.selectbox("Choose an event:", ["Baby Shower", "Marriage", "Reception", "Concert", "Funeral", "Birthday"])
text = st.text_area("Enter your prompt:")
prompt_text = f"""Create a formal invitation card for a {event} with the following specifications:

1. Theme description: {text+"greeting text for the {event} in black color, it must appear on the invitation card "}
2. Primary color scheme: {color}
3. Required elements:
   - Warm and praising invitation text should must be present on invitation card
   - Elegant opening greeting
   
Style guidelines:
- Use formal and sophisticated language
- Include at least 2-3 english sentences of heartfelt praise
- Maintain a celebratory and honored tone
- Incorporate the theme elements: {text}
- Ensure the text reflects the significance of the {event}

Please format the invitation with clear sections and appropriate spacing."""
generate_button = st.button("Generate Image")

if generate_button:
    try:
        st.write("Generating image... Please wait.")
        
        # Initialize the InferenceClient with your Hugging Face API token
        client = InferenceClient(
            api_key="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" # Replace with your Hugging Face token
        )
        
        # Start the timer
        start_time = time.time()
        
        # Generate an image using the text-to-image model
        image = client.text_to_image(
            prompt=prompt_text,
            model="stabilityai/stable-diffusion-3.5-large-turbo"
        )
        
        # Ensure the output is a valid image
        if not isinstance(image, Image.Image):
            raise TypeError("Generated output is not a valid image.")
        
        # Convert the image to bytes
        image_bytes = BytesIO()
        image.save(image_bytes, format="PNG")
        image_bytes.seek(0)
        
        # Display the image in Streamlit
        st.image(image, caption=f"Generated {event} Invitation Card", use_column_width=True)
        
        # Calculate the time taken
        end_time = time.time()
        elapsed_time = end_time - start_time
        st.write(f"Process took {elapsed_time:.2f} seconds.")
        
        # Provide a download button for the generated image
        st.download_button(label="Download Image", data=image_bytes, file_name=f"{event.lower().replace(' ', '_')}_invitation_card.png", mime="image/png")
        
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
