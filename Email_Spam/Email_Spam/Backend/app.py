
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib

app = FastAPI(
    title="Email Spam Detection API",
    description="FastAPI backend for Supervised Machine Learning Email Spam Detection",
    version="1.0.0"
)



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

try:
    model = joblib.load("spam_model.pkl")

    print("Spam detection model loaded successfully.")

except Exception as e:
    raise RuntimeError(
        f"Error loading spam_model.pkl: {e}"
    )

class EmailRequest(BaseModel):

    email_text: str = Field(
        ...,
        min_length=1,
        description="Email message to classify"
    )

@app.get("/")
def home():

    return {
        "status": "active",
        "message": "Email Spam Detection API is running"
    }


@app.post("/predict")
def predict_spam(data: EmailRequest):

    try:

        email_text = data.email_text.strip()

        if not email_text:

            raise HTTPException(
                status_code=400,
                detail="Email text cannot be empty."
            )

        prediction = model.predict([email_text])[0]

        if int(prediction) == 1:

            result = "Spam"

        else:

            result = "Not Spam"


        confidence = None

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(
                [email_text]
            )[0]

            confidence = round(
                float(max(probabilities)) * 100,
                2
            )

        return {
            "success": True,
            "prediction": int(prediction),
            "result": result,
            "confidence": confidence,
            "email_text": email_text
        }

    except HTTPException:

        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )