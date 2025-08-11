import os
import base64
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from openai import OpenAI
import requests
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv
from datetime import datetime, timedelta

# Load environment variables from .env file
load_dotenv()

# Initialize Flask
app = Flask(__name__)

# Configuration
GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")

ENDPOINT = "https://models.github.ai/inference"
MODEL = "openai/gpt-4.1"

# Initialize OpenAI client
client = OpenAI(
    base_url=ENDPOINT,
    api_key=GITHUB_TOKEN,
)

# In-memory conversation storage
# Structure: {phone_number: {"messages": [...], "last_active": datetime}}
conversations = {}
MAX_MESSAGES_PER_CONVERSATION = 200  # Generous limit for GPT-4.1's 1M token context

def get_conversation_history(phone_number):
    """Get conversation history for a phone number"""
    if phone_number not in conversations:
        conversations[phone_number] = {
            "messages": [{"role": "system", "content": """Eres un asistente de traducción inteligente especializado en traducir del inglés al español. Tu trabajo es:

1. **Traducir texto**: Cuando recibas texto en inglés, tradúcelo al español usando palabras simples y claras que cualquier persona pueda entender. Evita palabras complicadas o técnicas.

2. **Analizar imágenes**: Cuando recibas imágenes con texto, identifica y traduce todo el texto visible al español. También describe brevemente lo que ves en la imagen.

3. **Explicar de forma simple**: Tus explicaciones deben ser cortas, claras y fáciles de entender. No uses palabras rebuscadas ni hagas explicaciones muy largas.

4. **Responder en español**: Siempre responde en español, usando un tono amigable y profesional.

5. **Ayudar con el contexto**: Si algo no está claro, puedes pedir más contexto o dar una explicación breve del significado.

Ejemplo:
- Si recibo: "Hello, how are you?"
- Respondo: "Hola, ¿cómo estás?"

Mantén todo simple, claro y en español."""}],
            "last_active": datetime.now()
        }
    
    conversations[phone_number]["last_active"] = datetime.now()
    return conversations[phone_number]["messages"]

def add_to_conversation(phone_number, role, content):
    """Add a message to the conversation history"""
    messages = get_conversation_history(phone_number)
    messages.append({"role": role, "content": content})
    
    # Keep only the system message + last MAX_MESSAGES_PER_CONVERSATION messages
    if len(messages) > MAX_MESSAGES_PER_CONVERSATION + 1:  # +1 for system message
        # Keep system message and recent messages
        system_msg = messages[0]
        recent_messages = messages[-(MAX_MESSAGES_PER_CONVERSATION):]
        conversations[phone_number]["messages"] = [system_msg] + recent_messages

def analyze_image(media_url, caption="", phone_number=None):
    try:
        # Download the image with Twilio authentication
        response = requests.get(
            media_url, 
            auth=HTTPBasicAuth(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        )
        response.raise_for_status()
        
        # Convert image to base64
        image_base64 = base64.b64encode(response.content).decode('utf-8')
        
        # Determine image type from content-type header
        content_type = response.headers.get('content-type', 'image/jpeg')
        if 'image/' not in content_type:
            content_type = 'image/jpeg'  # Default assumption
        
        # Get conversation history
        messages = get_conversation_history(phone_number) if phone_number else []
        
        # Create the user message with image
        user_message = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Por favor analiza esta imagen y traduce al español cualquier texto que veas. También describe brevemente lo que hay en la imagen. {f'El usuario también envió este mensaje: {caption}' if caption else ''}"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{content_type};base64,{image_base64}"
                    }
                }
            ]
        }
        
        # Add user message to conversation
        if phone_number:
            add_to_conversation(phone_number, "user", f"[Envió una imagen{f' con mensaje: {caption}' if caption else ''}]")
        
        # Use conversation history + current image message
        current_messages = messages + [user_message]

        # Get GPT-4.1 vision analysis
        gpt_response = client.chat.completions.create(
            model=MODEL,
            messages=current_messages,
            temperature=0.3,  # Lower temperature for more consistent translations
            max_tokens=1000
        )
        
        assistant_reply = gpt_response.choices[0].message.content
        
        # Add assistant response to conversation
        if phone_number:
            add_to_conversation(phone_number, "assistant", assistant_reply)
        
        return assistant_reply
        
    except requests.RequestException as e:
        return f"⚠️ Error al descargar la imagen: {str(e)}"
    except Exception as e:
        return f"⚠️ Error al analizar la imagen: {str(e)}"

@app.route("/whatsapp", methods=['POST'])
def whatsapp_reply():
    # Initialize Twilio response
    resp = MessagingResponse()
    
    try:
        # Get the sender's phone number to track conversation
        from_number = request.form.get('From', '')
        
        # Check if there's an image
        num_media = int(request.form.get('NumMedia', 0))
        
        if num_media > 0:
            # Process image
            media_url = request.form.get('MediaUrl0')
            if not media_url:
                return str(resp.message("Lo siento, no pude acceder a la imagen. Por favor envíala de nuevo."))
                
            # Get caption if any
            caption = request.form.get('Body', '').strip()
            
            # Log for debugging
            print(f"Recibida imagen de {from_number}")
            print(f"Mensaje: {caption}")
            
            # Analyze image using GPT-4.1 vision with conversation memory
            reply_text = analyze_image(media_url, caption, from_number)
            
        else:
            # Process text message with conversation memory
            incoming_msg = request.form.get('Body', '').strip()
            if not incoming_msg:
                reply_text = "Recibí tu mensaje pero no pude entenderlo. Por favor envía texto o una imagen para traducir."
            else:
                # Get conversation history
                messages = get_conversation_history(from_number)
                
                # Add user message to conversation
                add_to_conversation(from_number, "user", incoming_msg)
                
                # Get updated messages including the new user message
                updated_messages = get_conversation_history(from_number)
                
                # Get GPT-4.1 response with conversation context
                try:
                    gpt_response = client.chat.completions.create(
                        model=MODEL,
                        messages=updated_messages,
                        temperature=0.3,  # Lower temperature for consistent translations
                        max_tokens=1000
                    )
                    reply_text = gpt_response.choices[0].message.content
                except Exception as api_error:
                    if "maximum context length" in str(api_error).lower():
                        # Token limit exceeded - trim conversation more aggressively
                        print(f"Límite de tokens excedido para {from_number}, recortando conversación")
                        
                        # Keep only system message + last 10 messages
                        system_msg = conversations[from_number]["messages"][0]
                        recent_messages = conversations[from_number]["messages"][-10:]
                        conversations[from_number]["messages"] = [system_msg] + recent_messages
                        
                        # Try again with trimmed conversation
                        trimmed_messages = conversations[from_number]["messages"]
                        gpt_response = client.chat.completions.create(
                            model=MODEL,
                            messages=trimmed_messages,
                            temperature=0.3,
                            max_tokens=1000
                        )
                        reply_text = gpt_response.choices[0].message.content
                    else:
                        # Different error
                        raise api_error
                
                # Add assistant response to conversation
                add_to_conversation(from_number, "assistant", reply_text)
                
        # Send the response
        resp.message(reply_text)
        
    except Exception as e:
        print(f"Error en whatsapp_reply: {str(e)}")
        resp.message(f"⚠️ Error procesando tu solicitud: {str(e)}")
    
    return str(resp)

@app.route("/", methods=['GET'])
def health_check():
    """Health check endpoint"""
    return "🤖 Servicio de Traducción AI funcionando correctamente 🌍"

@app.route("/stats", methods=['GET'])
def conversation_stats():
    """Endpoint to see conversation statistics (for debugging)"""
    stats = {
        "conversaciones_activas": len(conversations),
        "servicio": "Traductor AI Inglés-Español",
        "estado": "operativo"
    }
    return stats

if __name__ == "__main__":
    # Use Render's PORT environment variable (updated from Railway comment)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)