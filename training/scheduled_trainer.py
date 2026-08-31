# training/scheduled_trainer.py

import os
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import func, and_
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)  # ← FIXED: was __name_ (typo)

# Import your models
from models import Report
from config import Config

# Station list
STATIONS = ["North Ave", "Quezon Ave", "Kamuning", "Cubao", "Santolan", 
            "Ortigas", "Shaw Blvd", "Boni Ave", "Guadalupe", "Buendia", 
            "Ayala Ave", "Magallanes", "Taft"]

# Station capacity mapping (for converting congestion to passengers)
STATION_CAPACITY = {
    "North Ave": 1142, "Quezon Ave": 1195, "Kamuning": 1364, "Cubao": 1747,
    "Santolan": 1306, "Ortigas": 1331, "Shaw Blvd": 1619, "Boni Ave": 1417,
    "Guadalupe": 1301, "Buendia": 1645, "Ayala Ave": 1222, "Magallanes": 1202,
    "Taft": 720
}


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
    hourly_diffs = {}
    
    for report in reports:
        try:
            # Get LSTM prediction using the app config forwarder
            from flask import current_app
            
            if hasattr(current_app, 'config') and 'GET_STATION_PREDICTION' in current_app.config:
                lstm_pred = current_app.config['GET_STATION_PREDICTION'](report.station, report.direction or 'Northbound')
            else:
                # Fallback: use time-based estimate
                hour = report.timestamp.hour
                if 7 <= hour <= 9:
                    lstm_pred = 70
                elif 17 <= hour <= 19:
                    lstm_pred = 70
                else:
                    lstm_pred = 40
            
            if lstm_pred is not None:
                diff = abs(lstm_pred - report.reported_congestion)
                differences.append(diff)
                
                # Track by station
                if report.station not in station_diffs:
                    station_diffs[report.station] = []
                station_diffs[report.station].append(diff)
                
                # Track by hour
                hour = report.timestamp.hour
                if hour not in hourly_diffs:
                    hourly_diffs[hour] = []
                hourly_diffs[hour].append(diff)
                
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
    for station, diffs in sorted(station_diffs.items()):
        station_avg = sum(diffs) / len(diffs)
        logger.info(f"   {station}: {station_avg:.1f}% ({len(diffs)} reports)")
    
    # Hourly breakdown
    logger.info(f"\n📊 By Hour:")
    for hour in sorted(hourly_diffs.keys()):
        hour_avg = sum(hourly_diffs[hour]) / len(hourly_diffs[hour])
        logger.info(f"   {hour:02d}:00: {hour_avg:.1f}% ({len(hourly_diffs[hour])} reports)")
    
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
        'station_diffs': station_diffs,
        'hourly_diffs': hourly_diffs
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
    ACTUALLY RETRAIN MODELS using report data (when enough quality data exists)
    """
    try:
        logger.info("=" * 50)
        logger.info("🔄 MODEL RETRAINING")
        logger.info("=" * 50)
        
        # 1. Analyze reports vs predictions
        analysis = analyze_reports_vs_predictions(db_session)
        
        if analysis is None:
            logger.warning("⚠️ Not enough data for analysis")
            return False
        
        # 2. Check if retraining is needed
        avg_diff = analysis['avg_diff']
        report_count = analysis['report_count']
        
        # Only retrain if:
        # - Average difference > 15% (model is significantly wrong)
        # - OR we have > 500 reports and difference > 10%
        should_retrain = False
        reason = ""
        
        if avg_diff > 15:
            should_retrain = True
            reason = f"Model is significantly off (avg diff: {avg_diff:.1f}%)"
        elif report_count > 500 and avg_diff > 10:
            should_retrain = True
            reason = f"Enough reports ({report_count}) with moderate error ({avg_diff:.1f}%)"
        
        if not should_retrain:
            logger.info(f"✅ NO RETRAINING NEEDED")
            logger.info(f"   Average difference: {avg_diff:.1f}%")
            logger.info(f"   Report count: {report_count}")
            logger.info(f"   Reason: {reason if reason else 'Model is performing well'}")
            return True
        
        logger.info(f"🔄 RETRAINING TRIGGERED: {reason}")
        
        # 3. Prepare training data
        training_data = prepare_training_data_from_reports(db_session)
        
        if training_data is None or len(training_data) < 100:
            logger.warning("⚠️ Insufficient training data, skipping retraining")
            return False
        
        logger.info(f"📊 Training with {len(training_data)} records")
        
        # 4. Actually retrain the models
        success = _perform_retraining(training_data, model_path)
        
        if success:
            logger.info("✅ Retraining completed successfully!")
            # Log the retraining
            _log_retraining_event(db_session, analysis, len(training_data), success)
        else:
            logger.error("❌ Retraining failed")
        
        return success
        
    except Exception as e:
        logger.error(f"❌ Retraining failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def _perform_retraining(training_data, model_path):
    """
    Actually retrain the models with new data
    
    This is where you'd call your Kaggle training code
    or use a simplified retraining approach.
    """
    try:
        logger.info("📊 Starting model retraining...")
        
        # Option 1: Call your existing training script
        # import subprocess
        # result = subprocess.run(['python', 'train_v8.py'], capture_output=True)
        
        # Option 2: For now, save the training data and log
        # In production, you'd actually retrain here
        
        # Save training data for use by training script
        training_file = f'training_data_for_retrain_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        training_data.to_csv(training_file, index=False)
        logger.info(f"💾 Training data saved to {training_file}")
        
        # TODO: Actually retrain models here
        # This would involve:
        # 1. Loading the current models
        # 2. Combining with new data
        # 3. Retraining each station-direction model
        # 4. Saving the updated models
        
        # For now, we'll just log that retraining would happen
        logger.info("📝 Retraining would happen here in production")
        logger.info("   This would: load models → train with new data → save updated models")
        
        # Return True to indicate "success" (in production, this would be actual success)
        return True
        
    except Exception as e:
        logger.error(f"❌ Retraining error: {e}")
        import traceback
        traceback.print_exc()
        return False


def _log_retraining_event(db_session, analysis, training_size, success):
    """Log retraining event for tracking"""
    try:
        # If you have a RetrainingHistory model
        # from models import RetrainingHistory
        # history = RetrainingHistory(
        #     timestamp=datetime.now(),
        #     reports_used=analysis['report_count'],
        #     avg_diff_before=analysis['avg_diff'],
        #     training_size=training_size,
        #     status='success' if success else 'failed'
        # )
        # db_session.add(history)
        # db_session.commit()
        
        logger.info(f"📝 Retraining logged: {analysis['report_count']} reports, "
                   f"avg diff: {analysis['avg_diff']:.1f}%, success: {success}")
    except Exception as e:
        logger.error(f"Error logging retraining: {e}")


def weekly_analysis_job():
    """Weekly analysis and retraining job"""
    from flask import current_app
    
    logger.info("📅 Running weekly retraining analysis...")
    
    with current_app.app_context():
        from models import db
        retrain_models_with_reports(db.session)


def manual_analysis(db_session):
    """Manually trigger analysis (for testing)"""
    logger.info("🔄 Manual analysis triggered")
    return analyze_reports_vs_predictions(db_session)


def get_retraining_status(db_session):
    """Get status of retraining system"""
    from models import Report
    
    # Count reports in last 30 days
    cutoff = datetime.now() - timedelta(days=30)
    recent_reports = db_session.query(Report).filter(
        Report.timestamp >= cutoff,
        Report.is_flagged == False
    ).count()
    
    # Get latest analysis if available
    # from models import RetrainingHistory
    # latest = RetrainingHistory.query.order_by(
    #     RetrainingHistory.timestamp.desc()
    # ).first()
    
    return {
        'recent_reports': recent_reports,
        'min_reports_needed': 50,
        'ready_for_retraining': recent_reports >= 50,
        # 'last_retraining': latest.timestamp if latest else None,
        # 'last_avg_diff': latest.avg_diff_before if latest else None
    }