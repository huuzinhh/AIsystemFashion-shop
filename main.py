from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ProductsRecommendation import router as ai1_router
from ChatBot import router as ai2_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(ai1_router)
app.include_router(ai2_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
