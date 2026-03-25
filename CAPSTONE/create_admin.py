# check_db.py
from app import app, db, User
from datetime import datetime

with app.app_context():
    print("="*70)
    print("🔍 DATABASE DEBUG")
    print("="*70)
    
    # Create tables
    db.create_all()
    print("✅ Tables created")
    
    # Check existing users
    users = User.query.all()
    print(f"📊 Existing users: {len(users)}")
    
    for user in users:
        print(f"  • {user.username} - Role: {user.role}")
    
    # Create admin if none exists
    if len(users) == 0:
        print("\n📝 Creating admin user...")
        admin = User(
            username='admin@dotrmrt3.gov.ph',
            role='admin',
            created_at=datetime.now(),
            is_active=True
        )
        admin.password = 'admin123'
        
        db.session.add(admin)
        db.session.commit()
        print("✅ Admin user added to session and committed")
        
        # Verify it was saved
        new_admin = User.query.filter_by(username='admin@dotrmrt3.gov.ph').first()
        if new_admin:
            print(f"✅ Verification: {new_admin.username} - Role: {new_admin.role} - ID: {new_admin.id}")
        else:
            print("❌ Failed to save admin user!")
    else:
        print("✅ Admin already exists")
    
    # Final user list
    print("\n📊 Final users in database:")
    all_users = User.query.all()
    for user in all_users:
        print(f"  • {user.username} - Role: {user.role}")
    
    print("="*70)