"""
Security utilities for authentication and authorization.
"""

from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
from passlib.context import CryptContext
from pydantic import BaseModel

from multiagent.app.core.config import settings
from multiagent.app.schemas.auth import TokenData, UserBase

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a plain password against its hashed version.
    
    Args:
        plain_password: Plaintext password
        hashed_password: Hashed password to compare against
    
    Returns:
        Boolean indicating if password is correct
    """
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """
    Hash a password for storing.
    
    Args:
        password: Plaintext password
    
    Returns:
        Hashed password
    """
    return pwd_context.hash(password)

def create_access_token(
    data: dict, 
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    Create a JWT access token.
    
    Args:
        data: Dictionary of claims to encode in the token
        expires_delta: Optional duration for token expiration
    
    Returns:
        Encoded JWT token
    """
    to_encode = data.copy()
    
    # Set expiration
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        # Default expiration of 15 minutes
        expire = datetime.utcnow() + timedelta(minutes=15)
    
    to_encode.update({"exp": expire})
    
    # Encode JWT with secret key
    encoded_jwt = jwt.encode(
        to_encode, 
        settings.SECRET_KEY, 
        algorithm="HS256"
    )
    
    return encoded_jwt

def authenticate_user(username: str, password: str) -> Optional[UserBase]:
    """
    Authenticate a user by username and password.
    
    Args:
        username: User's username
        password: User's password
    
    Returns:
        Authenticated user or None
    """
    # TODO: Implement actual user retrieval from database
    # This is a placeholder implementation
    raise NotImplementedError("User authentication not implemented")

def get_current_user(token: str = Depends(oauth2_scheme)) -> UserBase:
    """
    Get the current user from a JWT token.
    
    Args:
        token: JWT access token
    
    Returns:
        Authenticated user
    
    Raises:
        HTTPException: If token is invalid or user cannot be retrieved
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        # Decode the JWT
        payload = jwt.decode(
            token, 
            settings.SECRET_KEY, 
            algorithms=["HS256"]
        )
        
        # Extract username
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        
        # Create token data
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    
    # TODO: Implement user retrieval from database
    # This is a placeholder implementation
    raise NotImplementedError("User retrieval not implemented")

def get_current_active_user(
    current_user: UserBase = Depends(get_current_user)
) -> UserBase:
    """
    Get the current active user.
    
    Args:
        current_user: User retrieved from token
    
    Returns:
        Active user
    
    Raises:
        HTTPException: If user is not active
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Inactive user"
        )
    
    return current_user