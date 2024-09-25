import os
import streamlit as st
from PIL import Image
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, AIMessage
import torch
from torchvision import models, transforms
import secrets
from dotenv import load_dotenv
from streamlit_option_menu import option_menu

# Load environment variables for the chatbot
load_dotenv()

# LLM setup
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# Create prompt template for chatbot
prompt = ChatPromptTemplate.from_messages([
    ("system", """ ."""), 
    MessagesPlaceholder(variable_name="chat_history"), 
    ("human", "{input}"), 
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# Search tool setup
search = TavilySearchResults()
search.name = "search"
search.description = "retrieve data which is not found in the vectorstore"
tools = [search]

# Create the agent
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools,
)

# Agent Executor
agentExecutor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)

# Define function to handle chat processing
def process_chat(agentExecutor, user_input, chat_history):
    response = agentExecutor.invoke({
        "input": user_input,
        "chat_history": chat_history
    })
    return response

# Function to load model for classification
def load_model(model_name):
    model = None
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=True)
        model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
        model.load_state_dict(torch.load(f'./models/{model_name}_fine_tuned.pth', map_location='cpu'))
    model.eval()  # Set to evaluation mode
    return model

# Preprocessing function for images
def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return preprocess(image).unsqueeze(0)  # Add batch dimension

# Streamlit navigation
with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options = ["Home", "Chatbot", "Kidney Condition Classification", "About"]
    )

if selected == "Home":
    # Home page content
    left_col, right_col = st.columns(2)
    
    # Right column content
    right_col.markdown("# Kidney Diagnosis and Support")
    right_col.markdown("### A tool with CT scan analysis and interactive chatbot support")
    right_col.markdown("**Created by Venu Gopal, Sai Shiva, Shishir**")
    
    # Left column content with a GIF
    left_col.image(r"1-cover.gif", caption="Kidney Diagnosis and Support", use_column_width=True)
    
    # Summary of the project
    st.markdown("---")  # This creates a horizontal line
    st.markdown("### Project Summary")
    st.markdown("""This project is designed to assist in the diagnosis of kidney conditions through analysis of CT scans. 
    It features an automated process for evaluating CT scan images to identify potential kidney issues, 
    along with an interactive chatbot that provides support and information regarding kidney health and 
    the diagnosis process. The goal is to enhance understanding of kidney health and offer timely assistance 
    for diagnosis, ultimately improving patient care.""")
    
    st.markdown("### Features")
    st.markdown("""- **CT Scan Analysis**: Automated evaluation of CT scans to identify abnormalities, such as kidney stones, tumors, or cysts. 
    This feature uses state-of-the-art algorithms to ensure high accuracy and reliability in diagnosis.
    
    - **Interactive Chatbot**: Engages users in conversation to provide tailored information and support. The chatbot can answer common questions about kidney health, risk factors, and lifestyle changes, making it a valuable resource for users.
    
    - **User-Friendly Interface**: Intuitive design that allows easy navigation and interaction. Users can upload scans, ask questions, and receive information seamlessly, ensuring a smooth experience.
    
    - **Secure Data Handling**: Ensures that user data is handled with the utmost confidentiality and security, complying with healthcare regulations.
    
    - **Educational Resources**: Provides access to articles, videos, and other resources about kidney health, empowering users to make informed decisions about their health.
    """)

    st.markdown("### Future Enhancements")
    st.markdown(""" Future updates to the project may include:
    
    - **Integration with Electronic Health Records (EHR)**: For seamless access to patient data, improving the diagnostic process and continuity of care.
    
    - **Enhanced Machine Learning Algorithms**: To improve the accuracy of scan analysis, allowing for the identification of more complex conditions.
    
    - **Support for Additional Medical Imaging Modalities**: Such as MRIs and ultrasounds, to broaden the tool's applicability.
    
    - **Personalized Health Recommendations**: Based on user input and analysis results, including diet, exercise, and preventive measures.
    
    - **Community Features**: A forum or chat feature where users can share experiences and support each other in managing kidney health.
    """)

elif selected == "Chatbot":
    st.title("MedGPT Chatbot")
    st.caption("Your intelligent healthcare assistant")

    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    with st.sidebar:
        if st.button("Clear Chat Window", use_container_width=True, type="primary"):
            st.session_state['chat_history'] = []
            st.experimental_rerun()

    for message in st.session_state['chat_history']:
        role = "assistant" if isinstance(message, AIMessage) else "user"
        with st.chat_message(role):
            st.markdown(message.content)

    if user_input := st.chat_input("You:"):
        user_input = user_input.strip()
        with st.chat_message("user"):
            st.markdown(user_input)

        st.session_state['chat_history'].append(HumanMessage(content=user_input))

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Thinking...")

            try:
                response = process_chat(agentExecutor, user_input, st.session_state['chat_history'])
                assistant_reply = response["output"].strip()
                st.session_state['chat_history'].append(AIMessage(content=assistant_reply))
                message_placeholder.markdown(assistant_reply)
            except Exception as e:
                message_placeholder.markdown(f"Error: {str(e)}")

elif selected == "Kidney Condition Classification":
    st.title("CT Kidney Condition Classification")

    classes = ['Cyst', 'Normal', 'Stone', 'Tumor']

    # Generate a CSRF token if it doesn't exist
    if 'csrf_token' not in st.session_state:
        st.session_state.csrf_token = secrets.token_hex(16)

    image_path = st.text_input("Enter the path to the image (e.g., C:\\path\\to\\image.jpg):")[1:-1]

    if image_path:
        if os.path.isfile(image_path):
            image = Image.open(image_path)
            st.image(image, caption='Uploaded Image.', use_column_width=True)

            if st.button("Classify"):
                st.write("Classifying...")
                model_name = 'resnet18'  # Change this as needed
                model = load_model(model_name)

                input_tensor = preprocess_image(image)
                input_tensor = input_tensor.to('cuda' if torch.cuda.is_available() else 'cpu')

                with torch.no_grad():
                    outputs = model(input_tensor)
                    _, preds = torch.max(outputs, 1)

                predicted_class = classes[preds.item()]
                st.markdown(f"<h2 style='color: dodgerblue; font-weight: bold; font-size:24px; color:#bbb;'>Predicted Condition: <span style='color:#fff;'>{predicted_class}</span></h2>", unsafe_allow_html=True)

                # Display information about the predicted condition
                if predicted_class == 'Cyst':
                    st.subheader("Kidney Cyst")
                    st.write("""A kidney cyst is a fluid-filled sac that forms in or on the kidney. Most kidney cysts are benign (non-cancerous) and do not cause any symptoms. They can sometimes lead to complications like infection or bleeding.""")
                    st.write("### Symptoms:")
                    st.write("""
                    - Often asymptomatic
                    - Pain in the side or back
                    - Fever or infection symptoms (if complications occur)
                    """)
                    st.write("### Precautions:")
                    st.write("""
                    - Regular monitoring through imaging tests (ultrasound or CT scans).
                    - Consult a healthcare provider if you experience any unusual symptoms.
                    - Maintain a healthy lifestyle to reduce kidney disease risk.
                    """)

                elif predicted_class == 'Stone':
                    st.subheader("Kidney Stone")
                    st.write("""Kidney stones are hard deposits made of minerals and salts that form inside your kidneys. They can cause severe pain and may require treatment to remove. Common symptoms include intense pain, nausea, and blood in the urine.""")
                    st.write("### Symptoms:")
                    st.write("""
                    - Severe pain in the back, side, lower abdomen, or groin
                    - Blood in urine
                    - Nausea and vomiting
                    - Frequent urination or urgency
                    """)
                    st.write("### Precautions:")
                    st.write("""
                    - Stay hydrated to help prevent stone formation.
                    - Limit salt and animal protein intake.
                    - Consult a healthcare provider about dietary adjustments based on stone type.
                    """)

                elif predicted_class == 'Tumor':
                    st.subheader("Kidney Tumor")
                    st.write("""A kidney tumor is an abnormal growth of cells in the kidney. Tumors can be benign or malignant (cancerous). Symptoms may include blood in the urine, persistent back pain, and unexplained weight loss.""")
                    st.write("### Symptoms:")
                    st.write("""
                    - Blood in urine
                    - Persistent back pain
                    - Unexplained weight loss
                    - Fatigue
                    - Swelling in the abdomen
                    """)
                    st.write("### Precautions:")
                    st.write("""
                    - Regular check-ups and imaging tests if you have risk factors.
                    - Maintain a healthy lifestyle with balanced diet and regular exercise.
                    - Avoid smoking and limit alcohol consumption.
                    - Discuss any family history of kidney cancer with your healthcare provider.
                    """)

                elif predicted_class == 'Normal':
                    st.subheader("Normal Kidney")
                    st.write("""Normal kidneys are healthy and function properly to filter blood, remove waste, and regulate electrolytes and fluids in the body. They play a vital role in maintaining overall health and homeostasis.""")
                    st.write("### Symptoms of Healthy Kidneys:")
                    st.write("""
                    - No pain or discomfort in the kidney area
                    - Normal urine output
                    - Absence of blood in urine
                    - No swelling in the legs or ankles
                    """)
                    st.write("### Precautions to Maintain Kidney Health:")
                    st.write("""
                    - Stay hydrated by drinking plenty of water.
                    - Maintain a balanced diet rich in fruits and vegetables.
                    - Regular exercise to promote overall health.
                    - Monitor blood pressure and blood sugar levels.
                    - Avoid excessive use of medications that can harm the kidneys (e.g., NSAIDs).
                    """)

        else:
            st.error("File not found. Please check the path and try again.")

elif selected == "About":
    st.title("About This Project")

    st.markdown("""
    ## Kidney Diagnosis and Support Tool

    This web application is designed to assist in the diagnosis of kidney conditions through CT scan analysis and an interactive chatbot. 

    ### Hosted on Streamlit
    The application is hosted on Streamlit, providing a user-friendly interface for interacting with advanced deep learning models and chatbot functionalities.

    ### Deep Learning Models
    The application employs several state-of-the-art deep learning architectures for classifying kidney conditions. The models include:
    - **ResNet18**
    - **MobileNetV2**
    - **EfficientNetB0**
    - **AlexNet**
    - **SqueezeNet**

    Each model has been fine-tuned on a dataset consisting of CT scan images labeled as Normal, Cyst, Stone, or Tumor. During the fine-tuning process, the models achieved impressive accuracy, demonstrating their effectiveness in distinguishing between these conditions.

    #### Model Performance
    - **ResNet18** achieved a validation accuracy of approximately **99.20%**.
    - **MobileNetV2** achieved a validation accuracy of approximately **99.68%**.
    - **EfficientNetB0** demonstrated a robust performance with a validation accuracy of **98.39%**.
    - **AlexNet** and **SqueezeNet** also contributed to the overall classification task with significant accuracy rates.

    The training involved rigorous preprocessing and augmentation techniques to ensure the models generalize well to unseen data.

    ### Chatbot Functionality
    The interactive chatbot is powered by:
    - **Gemini API**: Utilized for natural language understanding and response generation, providing users with accurate information regarding kidney health.
    - **Tavily API**: Integrated for fetching additional data that may not be available in the existing knowledge base.

    The chatbot serves as a virtual assistant, answering user inquiries about kidney conditions, risk factors, and preventive measures. 

    ### Conclusion
    This project aims to enhance understanding and awareness of kidney health, ultimately improving patient care through timely diagnosis and informative support.
    """)