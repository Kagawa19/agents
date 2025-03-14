"""
Authentication endpoints for user registration, login, and token management.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from multiagent.app.schemas.auth import Token, UserBase
from multiagent.app.core.security import (
    create_access_token,
    authenticate_user,
    get_current_active_user
)

# Create the router
router = APIRouter()

@router.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    OAuth2 compatible token login, get an access token for future requests.
    
    Args:
        form_data: Form data containing username and password
    
    Returns:
        Token with access token and token type
    """
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = create_access_token(
        data={"sub": user.username}
    )
    
    return {
        "access_token": access_token, 
        "token_type": "bearer"
    }

@router.post("/register", response_model=UserBase)
async def register_user(user_data: UserBase):
    """
    Register a new user.
    
    Args:
        user_data: User registration information
    
    Returns:
        Registered user details
    """
    # Implement user registration logic
    # This is a placeholder - you'll need to add actual user creation logic
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED, 
        detail="User registration not implemented"
    )

@router.get("/me", response_model=UserBase)
async def read_users_me(current_user: UserBase = Depends(get_current_active_user)):
    """
    Get the current logged-in user's information.
    
    Args:
        current_user: Authenticated user retrieved via dependency
    
    Returns:
        Current user's details
    """
    return current_user