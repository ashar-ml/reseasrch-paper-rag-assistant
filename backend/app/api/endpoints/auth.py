import json
import sqlite3
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from loguru import logger

from app.core.database import get_db_connection
from app.core.security import hash_password, verify_password

router = APIRouter()

class UserAuthSchema(BaseModel):
    username: str = Field(..., min_length=3, max_length=50, description="Unique account username.")
    password: str = Field(..., min_length=4, max_length=50, description="Account password.")

@router.post("/signup", status_code=status.HTTP_201_CREATED)
async def signup(payload: UserAuthSchema):
    """
    Registers a new user account if the username is unique.
    """
    logger.info(f"Received signup request for username: '{payload.username}'")
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Check if username is taken
        cursor.execute("SELECT id FROM users WHERE username = ?", (payload.username,))
        if cursor.fetchone():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username is already registered."
            )
            
        # Hash and store user
        hashed = hash_password(payload.password)
        cursor.execute(
            "INSERT INTO users (username, hashed_password) VALUES (?, ?)",
            (payload.username, hashed)
        )
        conn.commit()
        logger.info(f"User '{payload.username}' registered successfully.")
        
        return {
            "status": "success",
            "message": f"Account '{payload.username}' created successfully."
        }
    except sqlite3.IntegrityError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username is already registered."
        )
    finally:
        conn.close()

@router.post("/login")
async def login(payload: UserAuthSchema):
    """
    Verifies user credentials and logs them in.
    """
    logger.info(f"Received login request for username: '{payload.username}'")
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("SELECT hashed_password FROM users WHERE username = ?", (payload.username,))
        row = cursor.fetchone()
        
        if not row or not verify_password(payload.password, row[0]):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password."
            )
            
        logger.info(f"User '{payload.username}' authenticated successfully.")
        return {
            "status": "success",
            "message": "Login successful.",
            "username": payload.username
        }
    finally:
        conn.close()


class SaveHistoryRequest(BaseModel):
    username: str = Field(..., description="The username to save history for.")
    role: str = Field(..., description="Role: 'user' or 'assistant'.")
    content: str = Field(..., description="Message text content.")
    citations: list = Field(default=[], description="List of citation objects.")

@router.get("/history")
async def get_history(username: str):
    """
    Fetches chat history for the given username.
    """
    logger.info(f"Fetching chat history for user: '{username}'")
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "SELECT role, content, citations FROM chat_history WHERE username = ? ORDER BY created_at ASC",
            (username,)
        )
        rows = cursor.fetchall()
        history = []
        for row in rows:
            citations_list = []
            if row[2]:
                try:
                    citations_list = json.loads(row[2])
                except Exception:
                    pass
            history.append({
                "role": row[0],
                "content": row[1],
                "citations": citations_list
            })
        return {"status": "success", "history": history}
    except Exception as e:
        logger.error(f"Error fetching history: {e}")
        raise HTTPException(status_code=500, detail=f"Database query failed: {str(e)}")
    finally:
        conn.close()

@router.post("/history")
async def save_history(payload: SaveHistoryRequest):
    """
    Saves a message to the persistent chat history for the given user.
    """
    logger.info(f"Saving chat history message for user: '{payload.username}' ({payload.role})")
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        citations_str = json.dumps(payload.citations) if payload.citations else None
        cursor.execute(
            "INSERT INTO chat_history (username, role, content, citations) VALUES (?, ?, ?, ?)",
            (payload.username, payload.role, payload.content, citations_str)
        )
        conn.commit()
        return {"status": "success", "message": "Message saved to history."}
    except Exception as e:
        logger.error(f"Error saving history: {e}")
        raise HTTPException(status_code=500, detail=f"Database insert failed: {str(e)}")
    finally:
        conn.close()

@router.delete("/history")
async def clear_history(username: str):
    """
    Clears persistent chat history for the given username.
    """
    logger.info(f"Clearing chat history for user: '{username}'")
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM chat_history WHERE username = ?", (username,))
        conn.commit()
        return {"status": "success", "message": "Chat history cleared successfully."}
    except Exception as e:
        logger.error(f"Error clearing history: {e}")
        raise HTTPException(status_code=500, detail=f"Database delete failed: {str(e)}")
    finally:
        conn.close()
