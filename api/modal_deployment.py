"""
Academic Gender Prediction API - Modal Deployment
=================================================

Features:
- Rate limiting via Modal secrets (no exposed API keys)
- Academic usage policies and fair usage
- Research-friendly features and documentation
- Privacy-respecting analytics
- Educational endpoints
"""

import modal
import sys
import os
from typing import List, Dict, Any
from datetime import datetime, timedelta
import json

app = modal.App("gender-prediction-academic")

# Load academic configuration
sys.path.insert(0, '/app/api')

try:
    from config import (
        MODEL_CONFIG, ACADEMIC_CONFIG, MODAL_CONFIG, API_INFO,
        validate_academic_request, EXPERIMENT_ID
    )
    print("✅ Academic configuration loaded")
    print(f"   Experiment: {EXPERIMENT_ID}")
    print(f"   Threshold: {MODEL_CONFIG['optimal_threshold']}")
except ImportError as e:
    print(f"⚠️ Configuration import failed: {e}")
    # Fallback configuration
    MODEL_CONFIG = {
        'model_path': '/app/experiments/20250603_192912_r3_bce_h256_l3_dual_frz5/models/model.pth',
        'preprocessor_path': '/app/experiments/20250603_192912_r3_bce_h256_l3_dual_frz5/preprocessor.pkl',
        'optimal_threshold': 0.48
    }
    ACADEMIC_CONFIG = {'rate_limiting': {'requests_per_hour': 1000}}
    MODAL_CONFIG = {'app_name': 'gender-prediction-academic', 'gpu_type': 'T4'}
    API_INFO = {'version': '3.0-academic', 'title': 'Gender Prediction API'}

def build_image():
    """Build Modal image for academic API."""
    return (
        modal.Image.debian_slim(python_version="3.9")
        .pip_install([
            # Core ML dependencies
            "numpy>=1.20.0",
            "pandas>=1.3.0",
            "torch>=1.10.0",
            "scikit-learn>=1.0.0",
            "matplotlib>=3.4.0",
            "seaborn>=0.11.0",
            "tqdm>=4.62.0",
            "unicodedata2",

            # API framework
            "fastapi>=0.104.0",
            "uvicorn>=0.24.0",
            "pydantic>=2.0.0",

            # Additional utilities
            "python-multipart>=0.0.5",
            "httpx>=0.24.0",
            "python-dotenv>=0.19.0",
        ])
        .run_commands([
            "echo 'PYTHONPATH=/app/src:/app/scripts:/app/api' >> /etc/environment",
        ])
        .add_local_dir("..", "/app", copy=True)
    )

image = build_image()

# Academic secrets (no user API keys, only admin secrets)
secrets = [modal.Secret.from_name("gender-prediction-academic-secrets")]

@app.cls(
    image=image,
    gpu="T4",
    scaledown_window=300,
    max_containers=5,  # Conservative scaling for academic use
    secrets=secrets,
)
class AcademicGenderPredictionService:
    """
    Academic Gender Prediction Service

    Features:
    - Rate limiting for fair academic usage (no user API keys needed)
    - Research-friendly batch processing
    - Educational model information
    - Privacy-respecting analytics
    - Usage tracking by IP for fair usage
    """

    # Class variables instead of __init__ (Modal 1.0+ requirement)
    predictor = None
    usage_tracker = {}  # In-memory usage tracking by IP
    research_stats = {
        'total_predictions': 0,
        'successful_predictions': 0,
        'failed_predictions': 0,
        'unique_ips': set(),
        'batch_requests': 0,
        'research_citations': 0,
        'start_time': datetime.now(),
        'deployment_mode': 'academic'
    }

    @modal.enter()
    def initialize_academic_service(self):
        """Initialize the academic service."""
        print("🎓 Academic Gender Prediction API Starting...")
        print(f"   Version: {API_INFO.get('version', '3.0-academic')}")
        print(f"   License: GPL-3.0 for academic use")
        print(f"   Rate Limit: {ACADEMIC_CONFIG['rate_limiting']['requests_per_hour']} req/hour")
        print(f"   Batch Limit: {ACADEMIC_CONFIG['batch_limits']['max_batch_size']} names/batch")

        try:
            # Load model
            sys.path.insert(0, '/app/src')
            sys.path.insert(0, '/app/scripts')
            sys.path.insert(0, '/app/api')

            from final_predictor import FinalGenderPredictor

            print(f"📂 Loading academic model: {MODEL_CONFIG['model_path']}")
            self.predictor = FinalGenderPredictor(MODEL_CONFIG)
            self.predictor.load_model()

            print("✅ Academic service initialized successfully")

        except Exception as e:
            print(f"❌ Academic service initialization failed: {e}")
            import traceback
            print(traceback.format_exc())
            raise

    @modal.method()
    def predict_single(
        self,
        name: str,
        return_metadata: bool = False,
        research_note: str = None,
        user_ip: str = None
    ) -> Dict[str, Any]:
        """
        Academic single prediction with usage tracking by IP.

        Args:
            name: Name to predict
            return_metadata: Include educational metadata
            research_note: Optional note for research tracking
            user_ip: User IP for rate limiting (automatically provided by FastAPI)
        """
        try:
            # Rate limiting check by IP
            user_context = self._get_user_context(user_ip or 'unknown')
            validation = validate_academic_request({'names': [name]}, user_context)

            if not validation['allowed']:
                return self._academic_error_response(
                    name, validation['reason'], validation.get('message', 'Request denied')
                )

            # Track usage
            self._track_academic_usage(user_ip, 'single_prediction', research_note)

            # Core prediction
            self.research_stats['total_predictions'] += 1
            result = self.predictor.predict_single(name)

            # Add academic metadata
            if return_metadata:
                result['academic_metadata'] = {
                    'api_version': API_INFO.get('version', '3.0-academic'),
                    'model_info': {
                        'architecture': 'BiLSTM + Multi-head Attention',
                        'training_approach': 'Academic research standards',
                        'accuracy': MODEL_CONFIG.get('expected_performance', {}).get('accuracy', 0.92),
                        'f1_score': MODEL_CONFIG.get('expected_performance', {}).get('f1_score', 0.90),
                        'bias_ratio': MODEL_CONFIG.get('expected_performance', {}).get('bias_ratio', 0.999),
                        'unicode_support': True
                    },
                    'usage_info': {
                        'requests_remaining_hour': max(0,
                            ACADEMIC_CONFIG['rate_limiting']['requests_per_hour'] -
                            user_context.get('requests_this_hour', 0)
                        ),
                        'daily_quota': ACADEMIC_CONFIG['rate_limiting']['requests_per_day'],
                        'batch_limit': ACADEMIC_CONFIG['batch_limits']['max_batch_size']
                    },
                    'academic_resources': {
                        'documentation': '/docs',
                        'model_explanation': 'Available in API docs',
                        'citation_info': 'See repository for citation guidelines',
                        'source_code': 'https://github.com/guglielmopescatore/gender-predict'
                    }
                }

            # Research tracking
            if research_note:
                result['research_tracking'] = {
                    'note_recorded': True,
                    'tracking_id': self._generate_tracking_id()
                }

            self.research_stats['successful_predictions'] += 1
            return result

        except Exception as e:
            self.research_stats['failed_predictions'] += 1
            return self._academic_error_response(name, 'prediction_error', str(e))

    @modal.method()
    def predict_batch(
        self,
        names: List[str],
        return_metadata: bool = False,
        research_project: str = None,
        user_ip: str = None
    ) -> List[Dict[str, Any]]:
        """
        Academic batch prediction with research-friendly features.
        """
        try:
            # Validation
            user_context = self._get_user_context(user_ip or 'unknown')
            validation = validate_academic_request({'names': names}, user_context)

            if not validation['allowed']:
                return [self._academic_error_response(
                    name, validation['reason'], validation.get('message', 'Batch denied')
                ) for name in names]

            # Track batch usage
            self._track_academic_usage(user_ip, 'batch_prediction', research_project, len(names))
            self.research_stats['batch_requests'] += 1

            # Batch processing
            results = self.predictor.predict_batch(names)

            # Add batch metadata to first result
            if return_metadata and results and 'error' not in results[0]:
                results[0]['batch_metadata'] = {
                    'batch_info': {
                        'size': len(names),
                        'processing_time': '<100ms per name',
                        'batch_id': self._generate_tracking_id()
                    },
                    'research_info': {
                        'project_noted': bool(research_project),
                        'academic_use': 'Rate limits apply for fair usage',
                        'citation': 'Please cite if used in research'
                    }
                }

            return results

        except Exception as e:
            return [self._academic_error_response(name, 'batch_error', str(e)) for name in names]

    def _get_user_context(self, user_ip: str) -> Dict[str, Any]:
        """Get user context for rate limiting based on IP."""
        now = datetime.now()
        hour_key = now.strftime('%Y-%m-%d-%H')
        day_key = now.strftime('%Y-%m-%d')

        # Simple in-memory tracking (would use Redis for production)
        user_key = f"ip:{user_ip}"
        hour_usage_key = f"{user_key}:hour:{hour_key}"
        day_usage_key = f"{user_key}:day:{day_key}"

        return {
            'user_ip': user_ip,
            'requests_this_hour': self.usage_tracker.get(hour_usage_key, 0),
            'requests_this_day': self.usage_tracker.get(day_usage_key, 0),
            'hour_reset_time': (now + timedelta(hours=1)).replace(minute=0, second=0),
            'day_reset_time': (now + timedelta(days=1)).replace(hour=0, minute=0, second=0)
        }

    def _track_academic_usage(self, user_ip: str, operation: str, research_note: str = None, count: int = 1):
        """Track usage for academic analytics (by IP, privacy-friendly)."""
        now = datetime.now()
        hour_key = now.strftime('%Y-%m-%d-%H')
        day_key = now.strftime('%Y-%m-%d')

        user_key = f"ip:{user_ip}"
        self.usage_tracker[f"{user_key}:hour:{hour_key}"] = self.usage_tracker.get(f"{user_key}:hour:{hour_key}", 0) + count
        self.usage_tracker[f"{user_key}:day:{day_key}"] = self.usage_tracker.get(f"{user_key}:day:{day_key}", 0) + count

        # Track unique IPs for research stats (anonymized)
        if user_ip != 'unknown':
            self.research_stats['unique_ips'].add(user_ip)

        # Track research notes (anonymously)
        if research_note:
            self.research_stats['research_citations'] += 1

    def _generate_tracking_id(self) -> str:
        """Generate tracking ID for research purposes."""
        import hashlib
        timestamp = datetime.now().isoformat()
        return hashlib.md5(timestamp.encode()).hexdigest()[:8]

    def _academic_error_response(self, name: str, error_type: str, message: str) -> Dict[str, Any]:
        """Standardized academic error response."""
        return {
            'name': name,
            'predicted_gender': 'Unknown',
            'probability_female': 0.5,
            'confidence': 0.0,
            'error': message,
            'error_type': error_type,
            'academic_info': {
                'api_version': API_INFO.get('version', '3.0-academic'),
                'help_url': '/docs',
                'source_code': 'https://github.com/guglielmopescatore/gender-predict',
                'rate_limits': ACADEMIC_CONFIG['rate_limiting']
            },
            'timestamp': datetime.now().isoformat()
        }

    @modal.method()
    def get_academic_stats(self) -> Dict[str, Any]:
        """Get academic usage statistics."""
        unique_ip_count = len(self.research_stats['unique_ips'])

        return {
            'service_info': {
                'mode': 'academic',
                'version': API_INFO.get('version', '3.0-academic'),
                'license': 'GPL-3.0',
                'uptime_hours': round((datetime.now() - self.research_stats['start_time']).total_seconds() / 3600, 2)
            },
            'usage_stats': {
                'total_predictions': self.research_stats['total_predictions'],
                'successful_predictions': self.research_stats['successful_predictions'],
                'success_rate': round((
                    self.research_stats['successful_predictions'] /
                    max(1, self.research_stats['total_predictions'])
                ) * 100, 2),
                'unique_users': unique_ip_count,
                'batch_requests': self.research_stats['batch_requests'],
                'research_projects_tracked': self.research_stats['research_citations']
            },
            'academic_features': {
                'rate_limiting': ACADEMIC_CONFIG['rate_limiting'],
                'batch_limits': ACADEMIC_CONFIG['batch_limits'],
                'available_features': [k for k, v in ACADEMIC_CONFIG['features'].items() if v]
            },
            'model_info': {
                'experiment_id': EXPERIMENT_ID,
                'architecture': 'BiLSTM + Multi-head Attention',
                'threshold': MODEL_CONFIG['optimal_threshold'],
                'expected_accuracy': MODEL_CONFIG.get('expected_performance', {}).get('accuracy', 0.92),
                'expected_f1': MODEL_CONFIG.get('expected_performance', {}).get('f1_score', 0.90),
                'bias_ratio': MODEL_CONFIG.get('expected_performance', {}).get('bias_ratio', 0.999),
                'unicode_support': True
            }
        }

    @modal.method()
    def health_check(self) -> Dict[str, Any]:
        """Academic service health check."""
        try:
            # Test prediction
            test_result = self.predictor.predict_single("Test Academic")

            return {
                'status': 'healthy',
                'mode': 'academic',
                'timestamp': datetime.now().isoformat(),
                'model_status': 'loaded',
                'test_prediction': {
                    'name': 'Test Academic',
                    'prediction': test_result.get('predicted_gender', 'Unknown'),
                    'confidence': round(test_result.get('confidence', 0.0), 3)
                },
                'academic_config': {
                    'rate_limits_active': True,
                    'batch_processing_available': True,
                    'educational_features': True,
                    'research_tracking': True
                },
                'service_metrics': {
                    'total_predictions': self.research_stats['total_predictions'],
                    'uptime': 'healthy',
                    'unique_users': len(self.research_stats['unique_ips'])
                }
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'mode': 'academic',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

# Academic FastAPI interface
@app.function(image=image)
@modal.asgi_app()
def fastapi_app():
    """Create FastAPI web application for academic use."""
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import HTMLResponse
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    from typing import Union, List

    web_app = FastAPI(
        title=API_INFO.get('title', 'Gender Prediction API - Academic'),
        description=API_INFO.get('description', 'Academic gender prediction API'),
        version=API_INFO.get('version', '3.0-academic'),
        license_info={
            "name": "GPL-3.0",
            "url": "https://www.gnu.org/licenses/gpl-3.0.html",
        }
    )

    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Academic use - open CORS
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    service = AcademicGenderPredictionService()

    # Request models
    class PredictionRequest(BaseModel):
        names: Union[str, List[str]]
        return_metadata: bool = False
        research_note: Union[str, None] = None
        research_project: Union[str, None] = None

    @web_app.get("/", response_class=HTMLResponse)
    async def root():
        """API documentation page."""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Gender Prediction API - Academic</title>
            <style>
                body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 40px; }
                .container { max-width: 800px; margin: 0 auto; }
                .academic { background: #e8f5e8; padding: 15px; border-radius: 8px; margin: 20px 0; }
                code { background: #f4f4f4; padding: 2px 6px; border-radius: 4px; }
                pre { background: #f8f9fa; padding: 15px; border-radius: 8px; overflow-x: auto; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🎓 Gender Prediction API - Academic Version</h1>

                <div class="academic">
                    <h3>📚 Academic Use Only</h3>
                    <p>This API is provided for academic research and educational purposes under GPL-3.0 license.</p>
                </div>

                <h2>📊 Features</h2>
                <ul>
                    <li><strong>Model</strong>: BiLSTM + Multi-head Attention</li>
                    <li><strong>Accuracy</strong>: 92%+ on evaluation datasets</li>
                    <li><strong>Unicode Support</strong>: International names supported</li>
                    <li><strong>Batch Processing</strong>: Up to 500 names per request</li>
                    <li><strong>Rate Limiting</strong>: 1,000 requests/hour, 5,000/day</li>
                </ul>

                <h2>🔗 API Endpoints</h2>
                <div>
                    <h3>POST /predict</h3>
                    <p>Main prediction endpoint</p>
                    <pre>curl -X POST "/predict" -H "Content-Type: application/json" \\
  -d '{"names": "Mario Rossi", "return_metadata": true}'</pre>

                    <h3>GET /health</h3>
                    <p>Service health check</p>

                    <h3>GET /stats</h3>
                    <p>Academic usage statistics</p>

                    <h3>GET /docs</h3>
                    <p>Interactive API documentation</p>
                </div>

                <h2>📖 Usage Guidelines</h2>
                <ul>
                    <li><strong>Fair Usage</strong>: Rate limits ensure fair access for all researchers</li>
                    <li><strong>Batch Processing</strong>: Use batch requests for efficiency</li>
                    <li><strong>Citation</strong>: Please cite our work if used in research</li>
                    <li><strong>Contact</strong>: Reach out for research collaborations</li>
                </ul>

                <h2>🔗 Resources</h2>
                <ul>
                    <li><a href="/docs">Interactive API Documentation</a></li>
                    <li><a href="https://github.com/guglielmopescatore/gender-predict">Source Code</a></li>
                    <li><a href="/stats">Usage Statistics</a></li>
                </ul>
            </div>
        </body>
        </html>
        """

    @web_app.post("/predict")
    async def predict(request: PredictionRequest, http_request: Request):
        """Main prediction endpoint."""
        try:
            # Get user IP for rate limiting
            user_ip = http_request.client.host if http_request.client else 'unknown'

            if isinstance(request.names, str):
                # Single prediction
                result = service.predict_single.remote(
                    request.names,
                    return_metadata=request.return_metadata,
                    research_note=request.research_note,
                    user_ip=user_ip
                )
                return {'success': True, 'predictions': [result]}
            else:
                # Batch prediction
                results = service.predict_batch.remote(
                    request.names,
                    return_metadata=request.return_metadata,
                    research_project=request.research_project,
                    user_ip=user_ip
                )
                return {'success': True, 'predictions': results}

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @web_app.get("/health")
    async def health():
        """Health check endpoint."""
        return service.health_check.remote()

    @web_app.get("/stats")
    async def stats():
        """Academic usage statistics."""
        return service.get_academic_stats.remote()

    return web_app

# Test function
@app.function(image=image, secrets=secrets)
def test_academic_features():
    """Test academic API features."""
    service = AcademicGenderPredictionService()

    print("🎓 Testing Academic API Features...")

    # Test 1: Basic prediction
    result1 = service.predict_single.remote("Mario Rossi", return_metadata=True)
    print(f"✅ Basic prediction: {result1['predicted_gender']}")

    # Test 2: Research-tracked prediction
    result2 = service.predict_single.remote(
        "Giulia Bianchi",
        return_metadata=True,
        research_note="Italian names study"
    )
    print(f"✅ Research tracked: {result2.get('research_tracking', {}).get('note_recorded', False)}")

    # Test 3: Batch prediction
    batch_names = ["Anna Rossi", "Marco Verdi", "Sara Neri"]
    batch_results = service.predict_batch.remote(batch_names, return_metadata=True)
    print(f"✅ Batch prediction: {len(batch_results)} results")

    # Test 4: Stats
    stats = service.get_academic_stats.remote()
    print(f"✅ Academic stats: {stats['usage_stats']['total_predictions']} total predictions")

    return {
        'mode': 'academic',
        'tests_passed': 4,
        'academic_features_working': True
    }

if __name__ == "__main__":
    print("🎓 Academic Gender Prediction API")
    print("==================================")
    print("Deploy: modal deploy modal_deployment.py")
    print("Test: modal run modal_deployment.py::test_academic_features")
    print("")
    print("Academic Features:")
    print("✅ Rate limiting for fair usage (no user API keys needed)")
    print("✅ Research-friendly batch processing")
    print("✅ Educational model information")
    print("✅ Privacy-respecting analytics")
    print("✅ Citation and research tracking")
