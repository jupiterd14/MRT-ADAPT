# api_visualizer.py
"""
API Flow Visualizer for MRT-3 System
Run this to see interactive API documentation
"""

import json
from datetime import datetime

# Your API endpoints organized by flow
API_FLOW = {
    "1. System Initialization (Load Time)": {
        "order": 1,
        "description": "Happens when Flask starts, not user-triggered",
        "endpoints": [
            {
                "path": "🔄 Auto-loading",
                "method": "STARTUP",
                "description": "directional_models, directional_scalers, PER_DIRECTION_MAX load automatically",
                "files": ["services/model_loader.py", "services/feature_engineering.py"]
            }
        ]
    },
    
    "2. Authentication Flow": {
        "order": 2,
        "description": "User login and account management",
        "endpoints": [
            {
                "path": "/api/login/google",
                "method": "GET",
                "description": "Initiate Google OAuth login",
                "flow": "Browser → Google Login → Redirect to dashboard"
            },
            {
                "path": "/api/login/google/authorize",
                "method": "GET",
                "description": "Google OAuth callback handler",
                "flow": "Google → Flask → Create session → Redirect"
            },
            {
                "path": "/api/operator-signup",
                "method": "POST",
                "description": "Operator account creation from invite",
                "flow": "Invite link → Signup form → Create user → Redirect to login"
            }
        ]
    },
    
    "3. Admin Dashboard Flow": {
        "order": 3,
        "description": "Admin monitoring and management",
        "endpoints": [
            {
                "path": "/api/admin/dashboard-stats",
                "method": "GET",
                "description": "Fetch totals for dashboard cards",
                "flow": "Dashboard → API → Database → Return counts"
            },
            {
                "path": "/api/admin/station-status",
                "method": "GET",
                "description": "Get live congestion for all stations (both directions)",
                "flow": "Dashboard → API → predictor.py → LSTM models → Return congestion"
            },
            {
                "path": "/api/admin/operator-list",
                "method": "GET",
                "description": "List all operators with status",
                "flow": "Dashboard → API → Database → Return operator list"
            },
            {
                "path": "/api/admin/generate-invite",
                "method": "POST",
                "description": "Create operator invite link",
                "flow": "Admin form → API → Create user → Generate invite link"
            },
            {
                "path": "/api/admin/audit-log",
                "method": "GET",
                "description": "Fetch system activity log",
                "flow": "Dashboard → API → ActivityLog table → Return entries"
            },
            {
                "path": "/api/admin/flagged-actions",
                "method": "GET",
                "description": "Get flagged reports for review",
                "flow": "Dashboard → API → Reports table (flagged=True) → Return"
            }
        ]
    },
    
    "4. Model Performance Testing Flow": {
        "order": 4,
        "description": "ML model testing and validation",
        "endpoints": [
            {
                "path": "/api/model/run-auto-tests",
                "method": "POST",
                "description": "Batch test on random 2025 dates",
                "flow": "Click 'Run Tests' → API → Load 2025 data → Predict → Save results → Return summary"
            },
            {
                "path": "/api/model/predict/single",
                "method": "POST",
                "description": "Manual single prediction test",
                "flow": "Manual form → API → get_feature_sequence → LSTM predict → Return result"
            },
            {
                "path": "/api/model/performance/metrics",
                "method": "GET",
                "description": "Calculate MAE/MAPE from test results",
                "flow": "Dashboard → API → Read test_results CSVs → Calculate metrics → Return"
            },
            {
                "path": "/api/model/station-details",
                "method": "GET",
                "description": "Get detailed predictions for a station",
                "flow": "Click station row → API → Read station CSV → Return details"
            },
            {
                "path": "/api/model/chart/data",
                "method": "GET",
                "description": "Get prediction vs actual for charts",
                "flow": "Dashboard → API → Read test results → Return time series"
            },
            {
                "path": "/api/model/upload/batch",
                "method": "POST",
                "description": "Upload CSV test results",
                "flow": "Upload CSV → API → Validate → Save to test_results/ → Update metrics"
            },
            {
                "path": "/api/model/generate-predictions",
                "method": "POST",
                "description": "Generate fresh predictions on random dates",
                "flow": "API → Random date selection → LSTM predict → Save results"
            }
        ]
    },
    
    "5. Debug & Inspection Flow": {
        "order": 5,
        "description": "Development and troubleshooting endpoints",
        "endpoints": [
            {
                "path": "/api/health",
                "method": "GET",
                "description": "Check system health (models loaded, data files exist)",
                "flow": "Health check → Verify models → Return status"
            },
            {
                "path": "/api/debug/check-models",
                "method": "GET",
                "description": "Check which models are accessible",
                "flow": "Debug → List loaded models → Return"
            },
            {
                "path": "/api/debug/loaded-models",
                "method": "GET",
                "description": "List all loaded model keys",
                "flow": "Debug → Return directional_models.keys()"
            },
            {
                "path": "/api/debug/inspect-csv",
                "method": "GET",
                "description": "Inspect test_results CSV structure",
                "flow": "Debug → Read CSV files → Return structure"
            },
            {
                "path": "/api/debug/test-feature-sequence",
                "method": "GET",
                "description": "Test feature sequence generation",
                "flow": "Debug → Call get_feature_sequence_for_station → Return shape/range"
            }
        ]
    }
}


def print_flow_diagram():
    """Print ASCII flow diagram"""
    print("\n" + "=" * 80)
    print("🗺️  MRT-3 SYSTEM DATA FLOW DIAGRAM")
    print("=" * 80)
    
    print(r"""
    
    ┌─────────────────────────────────────────────────────────────────────────────────────┐
    │                              USER INTERFACE (Browser)                                 │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌───────────┐  │
    │  │   Admin     │  │  Operator   │  │  Commuter   │  │  Model Test │  │  Reports  │  │
    │  │  Dashboard  │  │  Dashboard  │  │    Map      │  │   Dashboard │  │   Page    │  │
    │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └─────┬─────┘  │
    └─────────┼────────────────┼────────────────┼────────────────┼───────────────┼────────┘
              │                │                │                │               │
              ▼                ▼                ▼                ▼               ▼
    ┌─────────────────────────────────────────────────────────────────────────────────────┐
    │                              API LAYER (Flask Routes)                                │
    │  /api/admin/*      /api/live/*       /api/reports/*    /api/model/*     /api/auth/*  │
    └─────────┬────────────────┴────────────────┴────────────────┴───────────────┬────────┘
              │                                                                  │
              ▼                                                                  ▼
    ┌─────────────────────────────────┐      ┌─────────────────────────────────────────┐
    │       SERVICE LAYER              │      │           DATABASE LAYER                 │
    │  ┌───────────────────────────┐  │      │  ┌─────────────────────────────────┐    │
    │  │  predictor.py             │  │      │  │  MySQL (SQLAlchemy)              │    │
    │  │  - get_directional_pred   │◄─┼──────┼──│  - User, Report, Broadcast       │    │
    │  │  - get_station_prediction │  │      │  │  - ActivityLog, Flagged reports   │    │
    │  └───────────┬───────────────┘  │      │  └─────────────────────────────────┘    │
    │              │                  │      │                                           │
    │  ┌───────────▼───────────────┐  │      │  ┌─────────────────────────────────┐    │
    │  │  feature_engineering.py   │  │      │  │  CSV Files (test_results/)       │    │
    │  │  - get_feature_sequence   │  │      │  │  - Station_direction_results.csv │    │
    │  │  - add_cyclical_features  │  │      │  │  - full_2025_test_*.csv          │    │
    │  └───────────┬───────────────┘  │      │  └─────────────────────────────────┘    │
    │              │                  │      │                                           │
    │  ┌───────────▼───────────────┐  │      │  ┌─────────────────────────────────┐    │
    │  │  model_loader.py          │  │      │  │  Model Files (models_2022-2024)   │    │
    │  │  - directional_models     │  │      │  │  - *.keras (LSTM models)         │    │
    │  │  - directional_scalers    │  │      │  │  - *.pkl (Scalers, max values)   │    │
    │  └───────────┬───────────────┘  │      │  └─────────────────────────────────┘    │
    │              │                  │      │                                           │
    │  ┌───────────▼───────────────┐  │      │                                           │
    │  │  2025 Data (CSV)          │  │      │                                           │
    │  │  - 2025.csv               │  │      │                                           │
    │  │  (passenger counts)       │  │      │                                           │
    │  └───────────────────────────┘  │      │                                           │
    └─────────────────────────────────┘      └─────────────────────────────────────────┘
    
    """)


def print_request_flow():
    """Print detailed request flow examples"""
    print("\n" + "=" * 80)
    print("🔄 DETAILED REQUEST FLOW EXAMPLES")
    print("=" * 80)
    
    flows = [
        {
            "name": "🏠 Admin Dashboard - Station Status",
            "flow": [
                "1. Browser loads admin dashboard -> calls loadStationStatus()",
                "2. JavaScript: fetch('/api/admin/station-status')",
                "3. Flask route: @admin_bp.route('/api/admin/station-status')",
                "4. Calls get_directional_prediction(station, direction) for each station",
                "5. predictor.py -> feature_engineering.py (builds 24h sequence)",
                "6. model_loader.directional_models[model_key].predict()",
                "7. target_scaler.inverse_transform() -> converts to percentage",
                "8. Returns JSON with congestion for all stations",
                "9. JavaScript updates table with status colors"
            ]
        },
        {
            "name": "🧪 Model Performance - Run Auto Tests",
            "flow": [
                "1. Admin clicks 'Run Tests' on Model Performance page",
                "2. JavaScript: fetch('/api/model/run-auto-tests')",
                "3. Flask route: @model_perf_bp.route('/model/run-auto-tests')",
                "4. Loads 2025.csv and adds all features",
                "5. For each station/direction, selects random dates (24h+ history)",
                "6. Calls get_directional_prediction() for each test",
                "7. Saves results to test_results/*.csv",
                "8. Returns summary (total tests, avg error, verdict counts)",
                "9. JavaScript refreshes metrics and chart"
            ]
        },
        {
            "name": "🔮 Single Prediction Test",
            "flow": [
                "1. User selects station, direction, datetime -> clicks 'Test Prediction'",
                "2. JavaScript: POST /api/model/predict/single",
                "3. Flask route: @model_perf_bp.route('/model/predict/single')",
                "4. Calls get_directional_prediction(station, direction, datetime)",
                "5. get_feature_sequence_for_station() -> builds 24h feature sequence",
                "6. Uses saved feature_scaler to normalize features",
                "7. directional_models[model_key].predict() -> raw output (0-1)",
                "8. target_scaler.inverse_transform() -> converts to percentage",
                "9. Also fetches actual congestion from 2025.csv for comparison",
                "10. Returns {predicted, actual, error}",
                "11. JavaScript displays result in modal"
            ]
        }
    ]
    
    for flow in flows:
        print(f"\n{flow['name']}")
        print("-" * 60)
        for step in flow['flow']:
            print(f"   {step}")


def print_service_hierarchy():
    """Print service layer dependencies"""
    print("\n" + "=" * 80)
    print("📚 SERVICE LAYER DEPENDENCIES")
    print("=" * 80)
    
    print(r"""
    
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                         api_model_performance.py                         │
    │  (Flask Routes - HTTP request handlers)                                 │
    └───────────────────────────────┬─────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                           predictor.py                                  │
    │  get_directional_prediction() - Main prediction logic                   │
    │  get_station_prediction() - Average both directions                     │
    │  get_fallback_directional_prediction() - When model fails               │
    └───────────────────────────────┬─────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                       feature_engineering.py                            │
    │  get_feature_sequence_for_station() - Builds 24h feature sequence       │
    │  add_cyclical_time_features() - Sin/cos transformations                 │
    │  add_smart_operating_flags() - Rush hour, pre/post opening flags        │
    │  load_data() - Loads and caches 2025.csv                                │
    └───────────────────────────────┬─────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                           model_loader.py                               │
    │  directional_models dict - {station_direction: Keras model}            │
    │  directional_scalers dict - {key: MinMaxScaler} for features/target     │
    │  PER_DIRECTION_MAX dict - Training max passengers for each direction    │
    └─────────────────────────────────────────────────────────────────────────┘
    
    """)


def generate_html_documentation():
    """Generate interactive HTML documentation"""
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MRT-3 API Documentation</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f5f5f5; padding: 20px; }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        h1 {{ color: #00224D; margin-bottom: 10px; }}
        .subtitle {{ color: #666; margin-bottom: 30px; }}
        .flow-section {{ background: white; border-radius: 12px; margin-bottom: 20px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        .flow-header {{ background: #00224D; color: white; padding: 15px 20px; cursor: pointer; }}
        .flow-header:hover {{ background: #003366; }}
        .flow-content {{ padding: 20px; display: none; }}
        .flow-content.active {{ display: block; }}
        .endpoint {{ border-left: 4px solid #3B82F6; margin-bottom: 15px; padding: 12px; background: #f8fafc; border-radius: 0 8px 8px 0; }}
        .endpoint-method {{ display: inline-block; padding: 3px 10px; border-radius: 20px; font-size: 11px; font-weight: bold; margin-right: 10px; }}
        .method-GET {{ background: #22C55E; color: white; }}
        .method-POST {{ background: #F59E0B; color: white; }}
        .method-DELETE {{ background: #EF4444; color: white; }}
        .method-PUT {{ background: #3B82F6; color: white; }}
        .endpoint-path {{ font-family: monospace; font-size: 14px; font-weight: bold; }}
        .endpoint-desc {{ margin-top: 8px; color: #555; font-size: 13px; }}
        .endpoint-flow {{ margin-top: 8px; font-size: 12px; color: #00224D; background: #EFF6FF; padding: 8px; border-radius: 6px; }}
        .stats {{ display: flex; gap: 20px; margin-bottom: 30px; flex-wrap: wrap; }}
        .stat-card {{ background: white; border-radius: 12px; padding: 20px; flex: 1; min-width: 150px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        .stat-number {{ font-size: 32px; font-weight: bold; color: #00224D; }}
        .stat-label {{ color: #666; margin-top: 5px; }}
        .flow-diagram {{ background: #1a1a2e; color: #eee; padding: 20px; border-radius: 12px; font-family: monospace; font-size: 12px; overflow-x: auto; margin-bottom: 30px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🚇 MRT-3 System API Documentation</h1>
        <div class="subtitle">Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
        
        <div class="stats">
            <div class="stat-card"><div class="stat-number">27</div><div class="stat-label">Total Endpoints</div></div>
            <div class="stat-card"><div class="stat-number">26</div><div class="stat-label">LSTM Models</div></div>
            <div class="stat-card"><div class="stat-number">13</div><div class="stat-label">Stations</div></div>
            <div class="stat-card"><div class="stat-number">2</div><div class="stat-label">Directions</div></div>
        </div>
        
        <div class="flow-diagram">
            <pre>
    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
    │   Browser   │────▶│   Flask     │────▶│  Service    │
    │   Request   │◀────│   Routes    │◀────│   Layer     │
    └─────────────┘     └─────────────┘     └─────────────┘
                              │                    │
                              ▼                    ▼
                       ┌─────────────┐     ┌─────────────┐
                       │  Database   │     │   Models    │
                       │  (MySQL)    │     │  (.keras)   │
                       └─────────────┘     └─────────────┘
            </pre>
        </div>
"""
    
    for section_name, section_data in API_FLOW.items():
        html += f"""
        <div class="flow-section">
            <div class="flow-header" onclick="toggleSection(this)">
                <strong>{section_name}</strong> - {section_data.get('description', '')}
            </div>
            <div class="flow-content">
        """
        
        for endpoint in section_data.get('endpoints', []):
            method_class = f"method-{endpoint['method'].replace(' ', '-')}"
            html += f"""
                <div class="endpoint">
                    <div><span class="endpoint-method {method_class}">{endpoint['method']}</span><span class="endpoint-path">{endpoint['path']}</span></div>
                    <div class="endpoint-desc">{endpoint['description']}</div>
                    <div class="endpoint-flow">📌 Flow: {endpoint['flow'] if 'flow' in endpoint else 'See console'}</div>
                    {f'<div class="endpoint-flow">📁 Files: {", ".join(endpoint.get("files", []))}</div>' if endpoint.get('files') else ''}
                </div>
            """
        
        html += """
            </div>
        </div>
        """
    
    html += """
    </div>
    <script>
        function toggleSection(header) {
            const content = header.nextElementSibling;
            content.classList.toggle('active');
        }
        // Open first section by default
        const firstContent = document.querySelector('.flow-content');
        if (firstContent) firstContent.classList.add('active');
    </script>
</body>
</html>
    """
    
    with open('api_documentation.html', 'w', encoding='utf-8') as f:
        f.write(html)
    print("\n✅ HTML documentation generated: api_documentation.html")


if __name__ == "__main__":
    print_flow_diagram()
    print_request_flow()
    print_service_hierarchy()
    generate_html_documentation()
    
    print("\n" + "=" * 80)
    print("✅ Documentation Generated!")
    print("=" * 80)
    print("\n📁 Files created:")
    print("   - api_documentation.html (open in browser)")
    print("\n💡 Study Order Recommendation:")
    print("   1. predictor.py (core prediction logic)")
    print("   2. feature_engineering.py (how features are built)")
    print("   3. model_loader.py (how models load)")
    print("   4. api_model_performance.py (API endpoints)")
    print("   5. admin_routes.py (dashboard integration)")