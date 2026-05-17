HOW TO RUN?

 1. CLONE REPOSITORY
git clone https://github.com/jupiterd14/MRT-ADAPT.git

cd MRT-ADAPT

----

2. Go to Capstone folder and create virtual environment
cd Capstone
python -m venv venv

-----

4. Activate venv (Windows)
.\venv\Scripts\activate

 (skip) For MAC users
source venv/bin/activate

-----

4. Install requirements
pip install -r requirements.txt

----

6. (skip) If requirements.txt not found, install manually:
pip install flask flask-sqlalchemy flask-login flask-oauthlib pandas numpy tensorflow python-dotenv werkzeug

-----

6 Run the app
python app.py
