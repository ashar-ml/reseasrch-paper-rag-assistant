from fastapi import APIRouter
from app.api.endpoints import upload, query, evaluation, auth

api_router = APIRouter()

api_router.include_router(upload.router, prefix="/upload", tags=["Upload & Processing"])
api_router.include_router(query.router, prefix="/query", tags=["Agentic Query"])
api_router.include_router(evaluation.router, prefix="/evaluate", tags=["RAGAS Evaluation"])
api_router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
