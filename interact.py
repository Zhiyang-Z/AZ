# app.py
from flask import Flask, jsonify, request, render_template
from gomoku import Env, State  # Your Gomoku environment



# Create instance
app = Flask(__name__)
# env = GomokuEnv()
# board = env.reset()

@app.route('/')
def index():
    return render_template('./gomoku.html')

@app.route('/reset', methods=['POST'])
def reset():
    global board
    board = env.reset()
    return jsonify({'board': board.tolist()})  # Convert numpy to list if needed

@app.route('/move', methods=['POST'])
def move():
    global board
    data = request.json
    print(data)
    action = data['action']  # Expected as [row, col]
    board, reward, done, info = env.step(action)
    return jsonify({
        'board': board.tolist(),
        'reward': reward,
        'done': done,
        'info': info
    })

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)