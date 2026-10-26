import joblib

model = joblib.load("spam_model.joblib")
print("SMS Spam Detector (type 'quit' to exit)")

while True:
    msg = input("\nEnter a message: ")
    if msg.strip().lower() == "quit":
        break
    if not msg.strip():
        continue
    label = model.predict([msg])[0]
    prob = model.predict_proba([msg])[0][1]
    print("Result:", "SPAM" if label == 1 else "NOT SPAM (ham)")
    print("Spam probability: {:.1f}%".format(prob * 100))