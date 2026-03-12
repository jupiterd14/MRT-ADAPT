# migrate_passwords.py
from app import app, db, User
from werkzeug.security import generate_password_hash

with app.app_context():
    users = User.query.all()
    for user in users:
        if user.password and not user.password_hash:
            # Migrate plain text to hash
            user.password_hash = generate_password_hash(user.password)
    db.session.commit()
    print("✅ Passwords migrated!")