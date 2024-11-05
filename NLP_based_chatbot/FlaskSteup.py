# from flask import Flask, request, jsonify
# from Final_Chatbot import get_response  # Import your chatbot function
#
# app = Flask(__name__)
#
# # Initialize dialogue state
# current_state = 'greeting'
#
# @app.route('/chat', methods=['POST'])
# def chatbot_response():
#     user_input = request.json.get("message")
#     response = get_response(current_state, user_input)  # Pass current state and user input
#     return jsonify({"response": response})
#
# if __name__ == '__main__':
#     app.run(debug=True)

# flask_app.py
from flask import Flask, request, jsonify
import Final_Chatbot  # Ensure this module is correctly referenced

app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    if user_message:
        # Capture the chatbot's response
        response = Final_Chatbot.get_response('greeting', user_message)  # Check the function parameters match
        return jsonify({'response': response})
    else:
        return jsonify({'error': 'No message received'}), 400  # Handle cases with no message

if __name__ == '__main__':
    app.run(debug=True)

