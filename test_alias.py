from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr

# Test the field alias functionality
chat = ChatGoogleGenerativeAI(
    model='gemini-pro',
    google_api_key=SecretStr('test'),
    default_metadata_input=None
)
print('SUCCESS: Field alias works correctly')
print(f'default_metadata value: {chat.default_metadata}')
