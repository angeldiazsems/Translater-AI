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
MAX_MESSAGES_PER_CONVERSATION = 200  # Much more generous limit for GPT-4.1's 1M token context

def get_conversation_history(phone_number):
    """Get conversation history for a phone number"""
    # Removed cleanup - conversations persist indefinitely
    
    if phone_number not in conversations:
        conversations[phone_number] = {
            "messages": [{"role": "system", "content": "You are a helpful assistant. You can analyze images and have conversations. Remember previous messages in this conversation."}],
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
                    "text": f"Please analyze this image and describe what you see. {f'The user also provided this caption: {caption}' if caption else ''}"
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
            add_to_conversation(phone_number, "user", f"[Sent an image{f' with caption: {caption}' if caption else ''}]")
        
        # Use conversation history + current image message
        current_messages = messages + [user_message]

        # Get GPT-4.1 vision analysis
        gpt_response = client.chat.completions.create(
            model=MODEL,
            messages=current_messages,
            temperature=0.7,
            max_tokens=1000
        )
        
        assistant_reply = gpt_response.choices[0].message.content
        
        # Add assistant response to conversation
        if phone_number:
            add_to_conversation(phone_number, "assistant", assistant_reply)
        
        return assistant_reply
        
    except requests.RequestException as e:
        return f"⚠️ Error downloading image: {str(e)}"
    except Exception as e:
        return f"⚠️ Error analyzing image: {str(e)}"

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
                return str(resp.message("Sorry, I couldn't access the image. Please try sending it again."))
                
            # Get caption if any
            caption = request.form.get('Body', '').strip()
            
            # Log for debugging
            print(f"Received image from {from_number}")
            print(f"Caption: {caption}")
            
            # Analyze image using GPT-4.1 vision with conversation memory
            reply_text = analyze_image(media_url, caption, from_number)
            
        else:
            # Process text message with conversation memory
            incoming_msg = request.form.get('Body', '').strip()
            if not incoming_msg:
                reply_text = "I received your message but couldn't understand it. Please send some text or an image."
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
                        temperature=0.7,
                        max_tokens=1000
                    )
                    reply_text = gpt_response.choices[0].message.content
                except Exception as api_error:
                    if "maximum context length" in str(api_error).lower():
                        # Token limit exceeded - trim conversation more aggressively
                        print(f"Token limit exceeded for {from_number}, trimming conversation")
                        
                        # Keep only system message + last 10 messages
                        system_msg = conversations[from_number]["messages"][0]
                        recent_messages = conversations[from_number]["messages"][-10:]
                        conversations[from_number]["messages"] = [system_msg] + recent_messages
                        
                        # Try again with trimmed conversation
                        trimmed_messages = conversations[from_number]["messages"]
                        gpt_response = client.chat.completions.create(
                            model=MODEL,
                            messages=trimmed_messages,
                            temperature=0.7,
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
        print(f"Error in whatsapp_reply: {str(e)}")
        resp.message(f"⚠️ Error processing your request: {str(e)}")
    
    return str(resp)

@app.route("/stats", methods=['GET'])
def conversation_stats():
    """Endpoint to see conversation statistics (for debugging)"""
    cleanup_old_conversations()
    stats = {
        "active_conversations": len(conversations),
        "conversations": {
            number: {
                "message_count": len(data["messages"]) - 1,  # Exclude system message
                "last_active": data["last_active"].strftime("%Y-%m-%d %H:%M:%S")
            }
            for number, data in conversations.items()
        }
    }
    return stats

if __name__ == "__main__":
    app.run(port=5000, debug=True)