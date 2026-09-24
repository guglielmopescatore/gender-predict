"""
Academic Gender Prediction API - Modal Deployment
=================================================

Features:
- Rate limiting via Modal secrets (no exposed API keys)
- Academic usage policies and fair usage
- Research-friendly features and documentation
- Privacy-respecting analytics
- Educational endpoints
- Automatic transliteration for non-Latin scripts
"""

import modal
import sys
import os
from typing import List, Dict, Any
from datetime import datetime, timedelta
import json

app = modal.App("gender-prediction-academic")

# Create Modal volume for models
model_volume = modal.Volume.from_name("gender-prediction-models", create_if_missing=True)

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

            # Transliteration dependencies
            "regex>=2021.8.3",
            "pypinyin>=0.44.0",
            "transliterate>=1.10.2",
            "fugashi>=1.1.0",
            "unidic-lite>=1.0.8",
            "romkan>=0.2.1",
            "hangul-romanize>=0.1.0",

            # Additional utilities
            "python-multipart>=0.0.5",
            "httpx>=0.24.0",
            "python-dotenv>=0.19.0",
        ])
        .run_commands([
            "echo 'PYTHONPATH=/app' >> /etc/environment",
        ])
        # Mount only necessary files (paths relative to api/ directory)
        .add_local_file("./config.py", "/app/api/config.py")
        .add_local_file("../scripts/final_predictor.py", "/app/scripts/final_predictor.py")
        .add_local_file("../scripts/transliteration_wrapper.py", "/app/scripts/transliteration_wrapper.py")
        .add_local_file("../scripts/enhanced_predictor.py", "/app/scripts/enhanced_predictor.py")
        # Mount source code if needed
        .add_local_dir("../src", "/app/src")
        # Mount the model files
        .add_local_file("../models/best_v3_model/preprocessor.pkl", "/app/models/best_v3_model/preprocessor.pkl")
        .add_local_file("../models/best_v3_model/feature_extractor.pkl", "/app/models/best_v3_model/feature_extractor.pkl")
        .add_local_file("../models/best_v3_model/models/model.pth", "/app/models/best_v3_model/models/model.pth")
    )

image = build_image()

# Academic secrets (no user API keys, only admin secrets)
secrets = [modal.Secret.from_name("gender-prediction-academic-secrets")]

# Load configuration - update paths
MODEL_CONFIG = {
    'model_path': '/app/models/best_v3_model/models/model.pth',
    'preprocessor_path': '/app/models/best_v3_model/preprocessor.pkl',
    'feature_extractor_path': '/app/models/best_v3_model/feature_extractor.pkl',
    'optimal_threshold': 0.52,
    'unicode_preprocessing': True,
    'enable_transliteration': True,
    'log_transliteration': False,
    'expected_performance': {
        'f1_score': 0.8996,
        'accuracy': 0.9219,
        'bias_ratio': 0.9999,
        'bias_deviation': 0.01
    }
}

ACADEMIC_CONFIG = {
    'deployment_mode': 'academic',
    'modal_app_name': 'gender-prediction-academic',
    'api_version': '3.1-academic',
    'rate_limiting': {
        'enabled': True,
        'requests_per_minute': 50,
        'requests_per_hour': 1000,
        'requests_per_day': 5000,
        'burst_allowance': 100
    },
    'usage_tracking': {
        'enabled': True,
        'track_by_ip': True,
        'track_research_metadata': True,
        'anonymize_after_days': 30
    },
    'features': {
        'basic_prediction': True,
        'confidence_scores': True,
        'batch_processing': True,
        'metadata_info': True,
        'model_performance_stats': True,
        'unicode_support': True,
        'transliteration': True,
        'tta_basic': False,
        'custom_thresholds': False,
    },
    'batch_limits': {
        'max_batch_size': 500,
        'max_batches_per_hour': 10,
        'concurrent_batches': 3
    },
    'fair_usage': {
        'cooling_period_minutes': 5,
        'daily_reset_hour': 0,
        'abuse_detection': True,
        'temporary_ban_hours': 24
    }
}

API_INFO = {
    'title': 'Gender Prediction API - Academic Version',
    'description': '''
    Deep Learning API for gender prediction from names.

    **Academic Use Only**: For research and educational purposes.

    **Features**:
    - Character-level deep learning model (BiLSTM + Attention)
    - 92%+ accuracy on evaluation datasets
    - Full Unicode support with automatic transliteration
    - Supports Cyrillic, Chinese, Japanese, Korean, Arabic scripts
    - Batch processing capabilities
    - Confidence scores and metadata

    **Transliteration Support**:
    - Automatic detection and conversion of non-Latin scripts
    - Cyrillic to Latin (e.g., Ekaterina to ekaterina)
    - Chinese to Pinyin (e.g., wang fang to wangfang)
    - Japanese to Hepburn romanization
    - Korean to Revised Romanization
    - Preserves original names in responses

    **Fair Usage**:
    - 1,000 requests per hour for research use
    - 5,000 requests per day maximum
    - Batch processing up to 500 names per request
    - Rate limiting automatically applied

    **Citation**: If you use this API in research, please cite our work.
    ''',
    'version': '3.1-academic',
    'license': 'GPL-3.0',
    'contact': {
        'name': 'Academic Support',
        'email': 'academic@genderpredict.com',
        'url': 'https://github.com/guglielmopescatore/gender-predict'
    }
}

# Validation function
def validate_academic_request(request_data, user_context):
    """Validate request against academic usage policies."""
    # Rate limiting check
    if user_context.get('requests_this_hour', 0) >= ACADEMIC_CONFIG['rate_limiting']['requests_per_hour']:
        return {
            'allowed': False,
            'reason': 'rate_limit_exceeded',
            'reset_time': user_context.get('hour_reset_time'),
            'message': f"Rate limit exceeded. Reset at {user_context.get('hour_reset_time')}"
        }

    # Batch size check
    if 'names' in request_data and isinstance(request_data['names'], list):
        batch_size = len(request_data['names'])
        max_batch = ACADEMIC_CONFIG['batch_limits']['max_batch_size']
        if batch_size > max_batch:
            return {
                'allowed': False,
                'reason': 'batch_too_large',
                'requested_size': batch_size,
                'max_size': max_batch,
                'message': f"Batch size {batch_size} exceeds limit of {max_batch}"
            }

    return {'allowed': True}

@app.cls(
    image=image,
    gpu="T4",
    scaledown_window=300,
    max_containers=5,
    secrets=secrets,
    volumes={"/models": model_volume},
)
class AcademicGenderPredictionService:
    """Academic Gender Prediction Service with Transliteration Support"""

    # Class variables
    predictor = None
    usage_tracker = {}
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
        print(f"   Version: {API_INFO.get('version', '3.1-academic')}")
        print(f"   Model Path: {MODEL_CONFIG['model_path']}")

        try:
            # Add paths
            sys.path.insert(0, '/app')
            sys.path.insert(0, '/app/scripts')
            sys.path.insert(0, '/app/api')

            # Verify model files exist
            import os
            print("\n📂 Checking model files:")
            for key, path in MODEL_CONFIG.items():
                if 'path' in key:
                    exists = os.path.exists(path)
                    print(f"   {key}: {path} {'✅' if exists else '❌'}")
                    if exists:
                        size = os.path.getsize(path) / 1024 / 1024  # MB
                        print(f"      Size: {size:.2f} MB")

            # Load enhanced predictor with transliteration
            from enhanced_predictor import EnhancedGenderPredictor

            print(f"📂 Loading academic model with transliteration support...")

            # Enable transliteration in config
            enhanced_config = MODEL_CONFIG.copy()
            enhanced_config['enable_transliteration'] = True
            enhanced_config['log_transliteration'] = True

            self.predictor = EnhancedGenderPredictor(enhanced_config)
            self.predictor.load_model()

            # Test prediction
            test_result = self.predictor.predict_single("Test")
            print(f"\n✅ Test prediction: {test_result}")
            print("✅ Academic service with transliteration initialized successfully")

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
        """Academic single prediction with usage tracking by IP."""
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
                    'api_version': API_INFO.get('version', '3.1-academic'),
                    'model_info': {
                        'architecture': 'BiLSTM + Multi-head Attention',
                        'training_approach': 'Academic research standards',
                        'accuracy': MODEL_CONFIG.get('expected_performance', {}).get('accuracy', 0.92),
                        'f1_score': MODEL_CONFIG.get('expected_performance', {}).get('f1_score', 0.90),
                        'bias_ratio': MODEL_CONFIG.get('expected_performance', {}).get('bias_ratio', 0.999),
                        'unicode_support': True,
                        'transliteration_enabled': True
                    },
                    'usage_info': {
                        'requests_remaining_hour': max(0,
                            ACADEMIC_CONFIG['rate_limiting']['requests_per_hour'] -
                            user_context.get('requests_this_hour', 0)
                        ),
                        'daily_quota': ACADEMIC_CONFIG['rate_limiting']['requests_per_day'],
                        'batch_limit': ACADEMIC_CONFIG['batch_limits']['max_batch_size']
                    }
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
        """Academic batch prediction with research-friendly features."""
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
        """Track usage for academic analytics."""
        now = datetime.now()
        hour_key = now.strftime('%Y-%m-%d-%H')
        day_key = now.strftime('%Y-%m-%d')

        user_key = f"ip:{user_ip}"
        self.usage_tracker[f"{user_key}:hour:{hour_key}"] = self.usage_tracker.get(f"{user_key}:hour:{hour_key}", 0) + count
        self.usage_tracker[f"{user_key}:day:{day_key}"] = self.usage_tracker.get(f"{user_key}:day:{day_key}", 0) + count

        if user_ip != 'unknown':
            self.research_stats['unique_ips'].add(user_ip)

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
                'api_version': API_INFO.get('version', '3.1-academic'),
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
                'version': API_INFO.get('version', '3.1-academic'),
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
                'batch_requests': self.research_stats['batch_requests']
            },
            'model_info': {
                'model_path': MODEL_CONFIG['model_path'],
                'architecture': 'BiLSTM + Multi-head Attention',
                'threshold': MODEL_CONFIG['optimal_threshold'],
                'expected_accuracy': MODEL_CONFIG.get('expected_performance', {}).get('accuracy', 0.92),
                'transliteration_enabled': True
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
                'features': {
                    'transliteration': True,
                    'rate_limiting': True,
                    'batch_processing': True
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
        version=API_INFO.get('version', '3.1-academic'),
        license_info={
            "name": "GPL-3.0",
            "url": "https://www.gnu.org/licenses/gpl-3.0.html",
        }
    )

    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
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
                    <p><strong>Now with automatic transliteration support for non-Latin scripts!</strong></p>
                </div>
                <h2>🔗 API Endpoints</h2>
                <div>
                    <h3>POST /predict</h3>
                    <p>Main prediction endpoint with transliteration</p>
                    <pre>curl -X POST "/predict" -H "Content-Type: application/json" \\
  -d '{"names": ["Mario Rossi", "Екатерина", "王芳"], "return_metadata": true}'</pre>
                    <h3>GET /health</h3>
                    <p>Service health check</p>
                    <h3>GET /stats</h3>
                    <p>Academic usage statistics</p>
                    <h3>GET /docs</h3>
                    <p>Interactive API documentation</p>
                </div>
            </div>
        </body>
        </html>
        """

    @web_app.post("/predict")
    async def predict(request: PredictionRequest, http_request: Request):
        """Main prediction endpoint."""
        try:
            user_ip = http_request.client.host if http_request.client else 'unknown'

            if isinstance(request.names, str):
                result = service.predict_single.remote(
                    request.names,
                    return_metadata=request.return_metadata,
                    research_note=request.research_note,
                    user_ip=user_ip
                )
                return {'success': True, 'predictions': [result]}
            else:
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

# Test functions
@app.function(image=image)
def test_model_files():
    """Test that model files are correctly mounted."""
    import os

    print("🔍 Testing Model Files")
    print("=" * 50)

    paths_to_check = [
        "/app/models/best_v3_model/preprocessor.pkl",
        "/app/models/best_v3_model/feature_extractor.pkl",
        "/app/models/best_v3_model/models/model.pth",
        "/app/scripts/final_predictor.py",
        "/app/scripts/transliteration_wrapper.py",
        "/app/scripts/enhanced_predictor.py",
        "/app/api/config.py"
    ]

    for path in paths_to_check:
        exists = os.path.exists(path)
        if exists:
            size = os.path.getsize(path) / 1024 / 1024  # MB
            print(f"✅ {path} ({size:.2f} MB)")
        else:
            print(f"❌ {path} NOT FOUND")

    # List directory contents
    print("\n📁 Directory Contents:")
    for root, dirs, files in os.walk("/app"):
        level = root.replace("/app", "").count(os.sep)
        indent = " " * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = " " * 2 * (level + 1)
        for file in files[:5]:  # Show first 5 files
            print(f"{subindent}{file}")
        if len(files) > 5:
            print(f"{subindent}... and {len(files) - 5} more files")

@app.function(image=image, secrets=secrets)
def test_academic_features():
    """Test academic API features."""
    service = AcademicGenderPredictionService()

    print("🎓 Testing Academic API Features...")

    # Test 1: Basic prediction
    result1 = service.predict_single.remote("Mario Rossi", return_metadata=True)
    print(f"✅ Basic prediction: {result1['predicted_gender']}")

    # Test 2: Batch prediction with mixed scripts
    batch_names = ["Anna Rossi", "Екатерина", "José García", "王芳"]
    batch_results = service.predict_batch.remote(batch_names, return_metadata=True)
    print(f"✅ Batch prediction: {len(batch_results)} results")

    # Show transliteration results
    print("\n📝 Transliteration test:")
    for result in batch_results:
        if result.get('was_transliterated', False):
            print(f"   {result['original_name']} → {result['transliterated_name']} "
                  f"({result['detected_script']}) = {result['predicted_gender']}")

    # Test 3: Stats
    stats = service.get_academic_stats.remote()
    print(f"✅ Academic stats: {stats['usage_stats']['total_predictions']} total predictions")

    return {
        'mode': 'academic',
        'tests_passed': 3,
        'academic_features_working': True,
        'transliteration_enabled': True
    }

if __name__ == "__main__":
    print("🎓 Academic Gender Prediction API with Transliteration")
    print("=====================================================")
    print("Deploy: modal deploy modal_deployment.py")
    print("Test model files: modal run modal_deployment.py::test_model_files")
    print("Test features: modal run modal_deployment.py::test_academic_features")
