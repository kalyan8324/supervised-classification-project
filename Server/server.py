from flask import *
app  = Flask(__name__)

@app.route('/login',methods = ['POST'])
def login():
    uname = request.args.get('uname')
    pwd = request.args.get('pwd')
    print('pwd hiiiiiiii')
    if uname == 'admin' and pwd == 'admin':
        return "hi this is kalyan from hyd"
    else:
        return "invalid username or password"


if __name__ == '__main__':
    app.run(debug = True,port=6611)
   