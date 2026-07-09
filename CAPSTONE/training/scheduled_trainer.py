# training/scheduled_trainer.py

import os
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta  # <-- ADD THIS IMPORT
from sqlalchemy import func, and_
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name_name__)

# Import your models
from models import Report
from config import Config

# Station list
STATIONS = ["North Ave", "Quezon Ave", "Kamuning", "Cubao", "Santolan", 
            "Ortigas", "Shaw Blvd", "Boni Ave", "Guadalupe", "Buendia", 
            "Ayala Ave", "Magallanes", "Taft"]

def analyze_reports_vs_predictions(db_session):
    """Compare reports against LSTM predictions - NO RETRAINING"""
    logger.info("📊 Analyzing report quality...")
    
    reports = db_session.query(Report).filter(
        Report.timestamp >= datetime.now() - timedelta(days=30),
        Report.is_flagged == False
    ).all()
    
    if len(reports) < 10:
        logger.info(f"⚠️ Not enough reports for analysis: {len(reports)} reports")
        return None
    
    differences = []
    station_diffs = {}
    
    for report in reports:
        try:
            # Get LSTM prediction (using your existing function)
            from services import get_station_prediction
            lstm_pred = get_station_prediction(report.station)
            
            if lstm_pred is not None:
                diff = abs(lstm_pred - report.reported_congestion)
                differences.append(diff)
                
                # Track by station
                if report.station not in station_diffs:
                    station_diffs[report.station] = []
                station_diffs[report.station].append(diff)
        except Exception as e:
            logger.error(f"Error processing report {report.id}: {e}")
    
    if not differences:
        logger.warning("No valid predictions to compare")
        return None
    
    avg_diff = sum(differences) / len(differences)
    max_diff = max(differences)
    min_diff = min(differences)
    
    logger.info(f"📊 Analysis Results:")
    logger.info(f"   Reports analyzed: {len(differences)}")
    logger.info(f"   Average difference: {avg_diff:.1f}%")
    logger.info(f"   Min difference: {min_diff:.1f}%")
    logger.info(f"   Max difference: {max_diff:.1f}%")
    
    # Station breakdown
    logger.info(f"\n📊 By Station:")
    for station, diffs in station_diffs.items():
        station_avg = sum(diffs) / len(diffs)
        logger.info(f"   {station}: {station_avg:.1f}% ({len(diffs)} reports)")
    
    # Recommendation
    if avg_diff < 10:
        logger.info("\n✅ LSTM predictions are very close to user reports!")
        logger.info("✅ No retraining needed - model is working well!")
    elif avg_diff < 20:
        logger.info("\n⚠️ LSTM predictions slightly off from user reports")
        logger.info("💡 Consider collecting more reports before retraining")
    else:
        logger.info("\n❌ LSTM predictions significantly differ from user reports")
        logger.info("💡 Investigate: Is LSTM wrong, or are users wrong?")
        logger.info("💡 Check for major events or operational changes")
    
    return {
        'avg_diff': avg_diff,
        'max_diff': max_diff,
        'min_diff': min_diff,
        'report_count': len(differences),
        'station_diffs': station_diffs
    }

def prepare_training_data_from_reports(db_session, days_back=30):
    """Prepare training data from reports for model retraining"""
    cutoff_date = datetime.now() - timedelta(days=days_back)
    
    reports = db_session.query(
        Report.station,
        Report.direction,
        Report.timestamp,
        Report.reported_congestion
    ).filter(
        and_(
            Report.timestamp >= cutoff_date,
            Report.is_flagged == False,
            Report.reported_congestion.isnot(None)
        )
    ).order_by(Report.timestamp.asc()).all()
    
    if len(reports) < 50:
        logger.warning(f"⚠️ Insufficient reports for retraining: {len(reports)} reports (need 50+)")
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame([(r.station, r.direction, r.timestamp, r.reported_congestion) 
                       for r in reports],
                      columns=['station', 'direction', 'timestamp', 'congestion'])
    
    # Extract time features
    df['hour'] = df['timestamp'].dt.hour
    df['weekday'] = df['timestamp'].dt.weekday
    df['month'] = df['timestamp'].dt.month
    df['day'] = df['timestamp'].dt.day
    df['hour_timestamp'] = df['timestamp'].dt.floor('h')
    
    # Group by station, direction, and hour
    grouped = df.groupby(['station', 'direction', 'hour_timestamp']).agg({
        'congestion': 'mean',
        'hour': 'first',
        'weekday': 'first',
        'month': 'first',
        'day': 'first'
    }).reset_index()
    
    logger.info(f"✅ Prepared {len(grouped)} training records from {len(reports)} reports")
    
    return grouped

def retrain_models_with_reports(db_session, model_path='./models_2022-2024_v8_20260616_081538/'):
    """
    DO NOT ACTUALLY RETRAIN - Just analyze and log
    
    This function only analyzes reports vs predictions.
    Actual retraining should only happen when data quality is sufficient.
    """
    try:
        logger.info("=" * 50)
        logger.info("📊 REPORT ANALYSIS (NO RETRAINING)")
        logger.info("=" * 50)
        
        # 1. Analyze reports vs predictions
        analysis = analyze_reports_vs_predictions(db_session)
        
        if analysis is None:
            logger.warning("⚠️ Not enough data for analysis")
            return False
        
        # 2. Prepare training data (just for logging)
        training_data = prepare_training_data_from_reports(db_session)
        
        if training_data is not None:
            # Save training data for reference
            training_file = f'training_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
            training_data.to_csv(training_file, index=False)
            logger.info(f"💾 Training data saved to {training_file} (for reference only)")
        
        # 3. Decision based on analysis
        if analysis['avg_diff'] < 10:
            logger.info("\n✅ RECOMMENDATION: NO RETRAINING NEEDED")
            logger.info(f"   Model accuracy: {100 - analysis['avg_diff']:.1f}%")
            return True
        elif analysis['avg_diff'] < 20:
            logger.info("\n⚠️ RECOMMENDATION: COLLECT MORE DATA")
            logger.info(f"   Current difference: {analysis['avg_diff']:.1f}%")
            logger.info("   Wait until you have 1000+ reports before retraining")
            return True
        else:
            logger.info("\n❌ RECOMMENDATION: INVESTIGATE")
            logger.info(f"   Large difference: {analysis['avg_diff']:.1f}%")
            logger.info("   Check for:")
            logger.info("   - Major operational changes")
            logger.info("   - Data quality issues")
            logger.info("   - Model degradation")
            return False
        
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def weekly_analysis_job():
    """Weekly analysis job - NO RETRAINING"""
    from flask import current_app
    
    logger.info("📅 Running weekly analysis...")
    
    with current_app.app_context():
        from models import db
        retrain_models_with_reports(db.session)

def manual_analysis(db_session):
    """Manually trigger analysis (for testing)"""
    logger.info("🔄 Manual analysis triggered")
    return analyze_reports_vs_predictions(db_session)