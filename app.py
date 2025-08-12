import os
import base64
import threading
import time
import requests as ping_requests
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
MAX_MESSAGES_PER_CONVERSATION = 300  # Increased from 200 for longer conversations

def get_conversation_history(phone_number):
    """Get conversation history for a phone number"""
    if phone_number not in conversations:
        conversations[phone_number] = {
            "messages": [{"role": "system", "content": """Eres un asistente inteligente y traductor especializado en inglés ↔ español. Tu trabajo es:

    Ayudar y traducir: Tu función principal es traducir del inglés al español usando palabras simples y claras que cualquiera pueda entender, pero también puedes responder y ayudar con cualquier otra pregunta o tarea que el usuario tenga.

    Analizar imágenes: Cuando recibas imágenes con texto, identifica y traduce todo el texto visible al español. También describe brevemente lo que ves en la imagen si es útil para el usuario.

    Explicar de forma simple: Mantén tus explicaciones cortas, claras y fáciles de entender. Evita palabras rebuscadas o explicaciones largas innecesarias.

    Responder en español: Siempre responde en español, con un tono amigable y profesional.

    Interpretar mensajes con errores: Si el mensaje tiene errores ortográficos o gramaticales, usa el contexto y mensajes previos para entenderlo.

    Mensajes incomprensibles: Si el mensaje es completamente incomprensible (por exceso de errores, incoherencia o ser puro “gibberish”), no inventes significado. Responde con algo breve como:

        "No entendí bien tu mensaje, ¿puedes escribirlo de otra forma?"

    Ayudar con el contexto: Si algo no está claro pero es parcialmente entendible, puedes pedir más contexto antes de responder o traducir.

Ejemplos:

    "Hello, how are you?" → "Hola, ¿cómo estás?"

    "hlp mi plz" (con contexto previo) → "Ayúdame, por favor"

    "sjd#@!!akd??" → "No entendí bien tu mensaje, ¿puedes escribirlo de otra forma?"

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

def send_twilio_message(to_number, message):
    """Send a message using Twilio client with better error handling"""
    try:
        from twilio.rest import Client
        from twilio.base.exceptions import TwilioRestException
        
        # Initialize Twilio client
        twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        
        # Send message
        twilio_client.messages.create(
            body=message,
            from_='whatsapp:+14155238886',  # Twilio Sandbox number
            to=to_number
        )
        
    except TwilioRestException as e:
        error_msg = f"Error enviando mensaje Twilio: {str(e)}"
        print(error_msg)
        
        # If it's a rate limit error, try to send a fallback message via webhook response
        if "429" in str(e) or "daily messages limit" in str(e).lower():
            print("⚠️ LÍMITE DIARIO DE TWILIO ALCANZADO - Usuario no recibirá respuesta")
            # We can't send another message since we're at the limit, but at least log it clearly
            
    except Exception as e:
        print(f"Error enviando mensaje: {str(e)}")

def process_translation_sync(from_number, incoming_msg):
    """Process translation synchronously and return result"""
    try:
        # Get conversation history
        messages = get_conversation_history(from_number)
        
        # Add user message to conversation
        add_to_conversation(from_number, "user", incoming_msg)
        
        # Get updated messages
        updated_messages = get_conversation_history(from_number)
        
        # Get GPT response (ONLY ONE API CALL)
        gpt_response = client.chat.completions.create(
            model=MODEL,
            messages=updated_messages,
            temperature=0.3,
            max_tokens=500,
            timeout=8  # Keep reasonable timeout for webhook response
        )
        reply_text = gpt_response.choices[0].message.content
        
        # Add assistant response to conversation
        add_to_conversation(from_number, "assistant", reply_text)
        
        return reply_text
        
    except Exception as api_error:
        if "timeout" in str(api_error).lower():
            return "⏱️ Traducción tomó mucho tiempo. Por favor intenta de nuevo."
        elif "maximum context length" in str(api_error).lower():
            # Handle token limit
            try:
                system_msg = conversations[from_number]["messages"][0]
                recent_messages = conversations[from_number]["messages"][-15:]
                conversations[from_number]["messages"] = [system_msg] + recent_messages
                
                trimmed_messages = conversations[from_number]["messages"]
                gpt_response = client.chat.completions.create(
                    model=MODEL,
                    messages=trimmed_messages,
                    temperature=0.3,
                    max_tokens=500,
                    timeout=8
                )
                reply_text = gpt_response.choices[0].message.content
                add_to_conversation(from_number, "assistant", reply_text)
                return reply_text
            except:
                return "⚠️ Error procesando mensaje. Intenta con un mensaje más corto."
        else:
            return "⚠️ Error procesando mensaje. Intenta de nuevo."

def analyze_image_sync(media_url, caption="", phone_number=None):
    """Process image analysis synchronously and return result"""
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
            content_type = 'image/jpeg'
        
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

        # Get GPT-4.1 vision analysis (ONLY ONE API CALL)
        gpt_response = client.chat.completions.create(
            model=MODEL,
            messages=current_messages,
            temperature=0.3,
            max_tokens=1000,
            timeout=12  # Slightly longer timeout for image processing
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
            # For images: Process and send single response
            media_url = request.form.get('MediaUrl0')
            caption = request.form.get('Body', '').strip()
            
            try:
                # Process image synchronously (with timeout protection)
                reply_text = analyze_image_sync(media_url, caption, from_number)
                resp.message(reply_text)
                
            except Exception as e:
                resp.message(f"⚠️ Error al procesar imagen: {str(e)}")
            
            return str(resp)
            
        else:
            # For text: Process and send single response
            incoming_msg = request.form.get('Body', '').strip()
            if not incoming_msg:
                resp.message("Recibí tu mensaje pero no pude entenderlo. Por favor envía texto o una imagen para traducir.")
                return str(resp)
            
            try:
                # Process translation synchronously (with timeout protection)
                reply_text = process_translation_sync(from_number, incoming_msg)
                resp.message(reply_text)
                
            except Exception as e:
                resp.message("⚠️ Error procesando mensaje. Intenta de nuevo.")
                print(f"Error en traducción: {str(e)}")
            
            return str(resp)
        
    except Exception as e:
        print(f"Error en whatsapp_reply: {str(e)}")
        resp.message("⚠️ Error temporal. Por favor intenta de nuevo.")
    
    return str(resp)

@app.route("/", methods=['GET'])
def health_check():
    """Health check endpoint"""
    return "🤖 Servicio de Traducción AI funcionando correctamente 🌍"

@app.route("/stats", methods=['GET'])
def conversation_stats():
    """Endpoint to see conversation statistics (for debugging)"""
    total_conversations = len(conversations)
    total_messages = 0
    memory_estimate_kb = 0
    
    for phone, data in conversations.items():
        msg_count = len(data["messages"]) - 1  # Exclude system message
        total_messages += msg_count
        # Rough estimate: ~100 bytes per message
        memory_estimate_kb += (msg_count * 0.1)  # KB
    
    stats = {
        "total conversations": total_conversations,
        "total messages": total_messages,
        "estimated memory(kb)": round(memory_estimate_kb, 2),
        "estimated memory(mb)": round(memory_estimate_kb / 1024, 3),
        "servicio": "Traductor AI Inglés-Español",
        "status": "operativo",
        "max_mensajes_por_conversacion": MAX_MESSAGES_PER_CONVERSATION,
    }
    return stats

def keep_alive():
    """Ping the app every 10 minutes to prevent sleeping"""
    while True:
        try:
            # Wait 10 minutes
            time.sleep(600)  # 600 seconds = 10 minutes
            
            # Ping your own health endpoint
            app_url = os.environ.get('RENDER_EXTERNAL_URL', 'https://whatsapp-bot-bx8f.onrender.com')
            ping_requests.get(f"{app_url}/", timeout=30)
            print("Keep-alive ping sent")
            
        except Exception as e:
            print(f"Keep-alive error: {str(e)}")

if not os.environ.get('WERKZEUG_RUN_MAIN'):  # Avoid running twice in debug mode
    # Start keep-alive thread
    keep_alive_thread = threading.Thread(target=keep_alive, daemon=True)
    keep_alive_thread.start()

if __name__ == "__main__":
    # Use Render's PORT environment variable 
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)