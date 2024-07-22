# Paperless Music Assistant
A sophisticated AI chatbot powered by the Zephyr 7B model, specializing in music-related information and creative tasks.
## Introduction
The Paperless Music Assistant is an advanced chatbot designed to provide comprehensive information on artists, albums, songs, genres, and musical history. It can also generate creative content like song lyrics, scripts, and stories based on music-related prompts. This assistant is powered by the Zephyr 7B model and utilizes Retrieval-Augmented Generation (RAG) to provide accurate and contextually relevant responses.
## Features
- Comprehensive music knowledge base
- Creative text generation for music-related content
- RAG-enhanced responses for improved accuracy
- User-friendly chat interface powered by Gradio
- Customizable chat parameters
## Prerequisites
Before setting up the Paperless Music Assistant, ensure you have:
- Python 3.7+
- pip (Python package manager)
- A Hugging Face account (for model access)
## Installation
1. Clone the repository:
   git clone https://github.com/yourusername/paperless-music-assistant.git
   cd paperless-music-assistant
2. Install the required packages:
   pip install -r requirements.txt
## Usage
1. Run the application:
   python app.py
2. Open your web browser and navigate to the local URL provided in the terminal.
3. Start interacting with the Paperless Music Assistant through the chat interface.
## How It Works
1. **PDF Processing**: The app loads and processes a PDF file ("Paperless Music.pdf") to build its knowledge base.
2. **Vector Database**: A vector database is created from the PDF content for efficient information retrieval.
3. **RAG System**: When a user asks a question, the system retrieves relevant information from the vector database.
4. **AI Response**: The Zephyr 7B model generates a response based on the user's query and the retrieved information.
## Customization
You can customize the assistant's behavior by modifying the following in `app.py`:
- `system_message`: Adjust the AI's persona and capabilities.
- Chat parameters: Modify `max_tokens`, `temperature`, and `top_p` for different response characteristics.
- `examples`: Add or modify example queries in the chat interface.
## Contributing
Contributions to improve the Paperless Music Assistant are welcome. Please follow these steps:
1. Fork the repository
2. Create a new branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request.
## Disclaimer
Paperless Music is a digital platform for music creation, distribution, and consumption. While we strive for accuracy and functionality, we cannot guarantee error-free operation or suitability for every user.
## Contact
For any questions or feedback regarding the Paperless Music Assistant, please contact Ni4368265@alphacollege.me.
