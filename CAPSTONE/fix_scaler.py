# test_predictions.py
from app import app
from datetime import datetime

def test_predictions():
    """Test predictions after scaler fix"""
    
    with app.app_context():
        # Import here to ensure app context is set
        from routes.api_predict import get_directional_prediction
        
        stations = ["North Ave", "Quezon Ave", "Kamuning", "Cubao", "Santolan", 
                    "Ortigas", "Shaw Blvd", "Boni Ave", "Guadalupe", "Buendia", 
                    "Ayala Ave", "Magallanes", "Taft"]
        
        now = datetime.now()
        
        print("\n" + "="*60)
        print("📊 TESTING PREDICTIONS AFTER SCALER FIX")
        print(f"⏰ Time: {now.strftime('%H:%M')}")
        print("="*60)
        print()
        
        total_north = 0
        total_south = 0
        
        for station in stations:
            north = get_directional_prediction(station, 'Northbound', now)
            south = get_directional_prediction(station, 'Southbound', now)
            avg = (north + south) / 2
            
            if avg > 80:
                status = "🔴 SEVERE"
            elif avg > 60:
                status = "🟠 CONGESTED"
            elif avg > 30:
                status = "🟡 MODERATE"
            else:
                status = "🟢 LIGHT"
            
            print(f"{station:15} | N: {north:5.1f}%  S: {south:5.1f}%  | {status}")
            total_north += north
            total_south += south
        
        print("\n" + "="*60)
        print(f"📊 Average Congestion: N: {total_north/len(stations):.1f}%  S: {total_south/len(stations):.1f}%")
        print("="*60)

if __name__ == '__main__':
    test_predictions()