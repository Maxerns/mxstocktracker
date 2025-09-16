import os
import html
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import Response
from lib.tracker import track_stocks
from lib.agent import handle_incoming_message, run_research_pipeline
import asyncio
from apscheduler.schedulers.background import BackgroundScheduler
from lib.stock_checker import get_stock_price
from lib.sms import send_sms
import sys
import json
from twilio.request_validator import RequestValidator
from dotenv import load_dotenv
load_dotenv()

# FASTAPI Configuration

app = FastAPI()

TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")  # Set this in your environment
TARGET_PHONE_NUMBER = os.getenv("TARGET_PHONE_NUMBER")  # Set this in your environment

WEBHOOK_URL = os.getenv("WEBHOOK_URL")  # This should be your public Twilio webhook URL

@app.post("/receive-message")
async def receive_message(request: Request):
  print("=== WEBHOOK RECEIVED ===")
  print(f"Headers: {dict(request.headers)}")
  
  signature = request.headers.get("X-Twilio-Signature", "")
  form = await request.form()
  print(f"Form data: {dict(form)}")
  
  validator = RequestValidator(TWILIO_AUTH_TOKEN)
  params = dict(form)
  
  if not validator.validate(WEBHOOK_URL, params, signature):
    print("❌ Signature validation failed!")
    return Response(content="Invalid signature", status_code=403)

  from_number = form.get("From", "")
  to_number = form.get("To", "")
  body = form.get("Body", "")
  
  print(f"From: {from_number}, To: {to_number}, Body: {body}")

  if (to_number != os.getenv("TWILIO_PHONE_NUMBER")):
    print("❌ Wrong to_number")
    return Response(content="Unauthorized", status_code=403)

  if (from_number != TARGET_PHONE_NUMBER):
    print("❌ Wrong from_number") 
    return Response(content="Unauthorized", status_code=403)

  safe_body = html.escape(body)
  print(f"Processing message: {safe_body}")

  try:
    # Process the message
    response_text = await handle_incoming_message(safe_body)
    print(f"Agent response: {response_text}")
    
    # Send SMS response
    send_sms(response_text)
    print("✅ SMS sent successfully")
    
    # IMPORTANT: Return proper response to Twilio
    return Response(content="OK", status_code=200)
    
  except Exception as e:
    print(f"❌ Error processing message: {e}")
    return Response(content="Error processing message", status_code=500)

# For testing purposes only

async def chat_terminal():
  print("Chat mode activated. Type 'exit' to quit.")
  while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
      print("Exiting chat.")
      break
    response = await handle_incoming_message(user_input)
    print(f"Bot: {response}")

# Run the project

if __name__ == "__main__":

  # Check to make sure the alert_history.json and tracker_list.json exist, otherwise create them

  if not os.path.exists("resources"):
    os.makedirs("resources")

  if not os.path.exists("resources/alert_history.json"):
    with open("resources/alert_history.json", "w") as f:
      json.dump({}, f)

  if not os.path.exists("resources/tracker_list.json"):
    with open("resources/tracker_list.json", "w") as f:
      json.dump([], f)

  if "-test" in sys.argv:
    if "-research" in sys.argv:
      stock_symbol = sys.argv[sys.argv.index("-research") + 1]

      stock_price = get_stock_price(stock_symbol)

      asyncio.run(run_research_pipeline(stock_symbol, stock_price.current_price, stock_price.previous_close))
    else:
      # CRON Job for tracking stock prices

      scheduler = BackgroundScheduler()
      scheduler.add_job(track_stocks, 'interval', minutes=1)
      scheduler.start()

      asyncio.run(chat_terminal())
  else:
    # CRON Job for tracking stock prices

    scheduler = BackgroundScheduler()
    scheduler.add_job(track_stocks, 'interval', minutes=15)
    scheduler.start()

    # Run with: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
    uvicorn.run(
      "main:app",
      host=os.getenv("HOST", "0.0.0.0"),
      port=int(os.getenv("PORT", "8000")),
      reload=True,
    )
