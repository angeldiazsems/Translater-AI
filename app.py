import os
import base64
import threading
import time
import whisper
import requests as ping_requests
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from openai import OpenAI
import requests
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv
from datetime import datetime, timedelta
import tempfile

# Load environment variables from .env file
load_dotenv()

# Initialize Flask
app = Flask(__name__)

# Configuration
GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")

ENDPOINT = "https://models.github.ai/inference"
MODEL = "gpt-4o-mini"

# Initialize OpenAI client
client = OpenAI(
    base_url=ENDPOINT,
    api_key=GITHUB_TOKEN,
)
whisper_model = whisper.load_model("small")

# In-memory conversation storage
# Structure: {phone_number: {"messages": [...], "last_active": datetime}}
conversations = {}
MAX_MESSAGES_PER_CONVERSATION = 300  # Increased from 200 for longer conversations

def get_conversation_history(phone_number):
    """Get conversation history for a phone number"""
    if phone_number not in conversations:
        conversations[phone_number] = {
            "messages": [{"role": "system", "content": "Eres un asistente útil. Si te preguntan algo en ingles, tradúcelo al español. Pero si no, contesta a la pregunta en español."}],
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
        print(f"🔄 Iniciando traducción para: {from_number[:10]}... | Mensaje: {incoming_msg[:50]}...")
        
        # Get conversation history
        messages = get_conversation_history(from_number)
        print(f"📚 Historial obtenido: {len(messages)} mensajes")
        
        # Add user message to conversation
        add_to_conversation(from_number, "user", incoming_msg)
        
        # Get updated messages
        updated_messages = get_conversation_history(from_number)
        print(f"🔤 Preparando llamada API con {len(updated_messages)} mensajes")
        
        # Get GPT response (ONLY ONE API CALL)
        print("🤖 Llamando a GitHub Models API...")
        gpt_response = client.chat.completions.create(
            model=MODEL,
            messages=updated_messages,
            temperature=0.3,
            max_tokens=1000,
            timeout=12 # Keep reasonable timeout for webhook response
        )
        reply_text = gpt_response.choices[0].message.content
        print(f"✅ Respuesta recibida: {reply_text[:100]}...")
        
        # Add assistant response to conversation
        add_to_conversation(from_number, "assistant", reply_text)
        
        return reply_text
        
    except Exception as api_error:
        error_msg = f"Error específico: {str(api_error)} | Tipo: {type(api_error).__name__}"
        print(f"❌ {error_msg}")
        
        if "timeout" in str(api_error).lower():
            return "⏱️ Traducción tomó mucho tiempo. Por favor intenta de nuevo."
        elif "maximum context length" in str(api_error).lower():
            print("🔄 Intentando con contexto reducido...")
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
                    max_tokens=1000,
                    timeout=12
                )
                reply_text = gpt_response.choices[0].message.content
                add_to_conversation(from_number, "assistant", reply_text)
                return reply_text
            except Exception as retry_error:
                print(f"❌ Error en retry: {str(retry_error)}")
                return "⚠️ Error procesando mensaje. Intenta con un mensaje más corto."
        else:
            return f"⚠️ Error: {str(api_error)[:100]}"  # Return actual error for debugging

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
            max_tokens=1200,
            timeout=15  # Slightly longer timeout for image processing
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
    
def process_voice_memo_sync(media_url, phone_number=None):
    """Process voice memo using FREE local Whisper model"""
    try:
        # Download the voice file
        response = requests.get(media_url, auth=HTTPBasicAuth(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN))
        
        # Save it temporarily  
        with tempfile.NamedTemporaryFile(delete=False, suffix=".ogg") as temp_audio:
            temp_audio.write(response.content)
            temp_audio_path = temp_audio.name
        
        try:
            # Transcribe with FREE Whisper
            result = whisper_model.transcribe(temp_audio_path)
            transcribed_text = result["text"].strip()
            
            # Add to conversation
            if phone_number:
                add_to_conversation(phone_number, "user", f"[Nota de voz]: {transcribed_text}")
            
            # Translate using your existing system
            messages = get_conversation_history(phone_number)
            messages.append({"role": "user", "content": transcribed_text})
            
            gpt_response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.3,
                max_tokens=500,
                timeout=12
            )
            
            reply_text = gpt_response.choices[0].message.content
            
            if phone_number:
                add_to_conversation(phone_number, "assistant", reply_text)
            
            return f"🎤 Escuché: _{transcribed_text}_\n\n📝 {reply_text}"
            
        finally:
            os.unlink(temp_audio_path)  # Delete temp file
                
    except Exception as e:
        return f"⚠️ Error con nota de voz: {str(e)}"



@app.route("/whatsapp", methods=['POST'])
def whatsapp_reply():
    # Initialize Twilio response
    resp = MessagingResponse()
    
    try:
        # Get the sender's phone number to track conversation
        from_number = request.form.get('From', '')
        
        # Check if there's media
        num_media = int(request.form.get('NumMedia', 0))
        
        if num_media > 0:
            # Get media info
            media_url = request.form.get('MediaUrl0')
            media_content_type = request.form.get('MediaContentType0', '')
            caption = request.form.get('Body', '').strip()
            
            print(f"📱 Media recibido - Tipo: {media_content_type}")
            
            try:
                # Check if it's a voice memo
                if media_content_type.startswith('audio/'):
                    print("🎤 Procesando nota de voz...")
                    reply_text = process_voice_memo_sync(media_url, from_number)
                
                # Check if it's an image
                elif media_content_type.startswith('image/'):
                    print("🖼️ Procesando imagen...")
                    reply_text = analyze_image_sync(media_url, caption, from_number)
                
                else:
                    reply_text = f"⚠️ Tipo de archivo no soportado: {media_content_type}. Envía imágenes o notas de voz."
                
                resp.message(reply_text)
                
            except Exception as e:
                resp.message(f"⚠️ Error al procesar archivo multimedia: {str(e)}")
            
            return str(resp)
            
        else:
            # For text: Process and send single response (keep your existing text logic)
            incoming_msg = request.form.get('Body', '').strip()
            if not incoming_msg:
                resp.message("Recibí tu mensaje pero no pude entenderlo. Por favor envía texto, imagen o nota de voz para traducir.")
                return str(resp)
            
            try:
                # Process translation synchronously (with timeout protection)
                reply_text = process_translation_sync(from_number, incoming_msg)
                resp.message(reply_text)
                
            except Exception as e:
                error_details = f"Error en traducción: {str(e)} | Tipo: {type(e).__name__}"
                print(error_details)
                resp.message(f"⚠️ Error procesando mensaje: {str(e)[:100]}")
            
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
        "conversaciones_totales": total_conversations,
        "total_mensajes": total_messages,
        "memoria_estimada_kb": round(memory_estimate_kb, 2),
        "memoria_estimada_mb": round(memory_estimate_kb / 1024, 3),
        "servicio": "Traductor AI Inglés-Español",
        "estado": "operativo",
        "modo": "Respuesta única - Sin loading messages",
        "max_mensajes_por_conversacion": MAX_MESSAGES_PER_CONVERSATION,
        "twilio_limite": "⚠️ Sandbox: 9 mensajes/día - 1 mensaje por traducción",
        "eficiencia": "9 traducciones completas por día",
        "nota": "Sin limpieza automática - memoria mínima para 1-2 usuarios"
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