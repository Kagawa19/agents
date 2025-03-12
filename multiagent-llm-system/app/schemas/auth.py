from typing import Optional

from pydantic import BaseModel, EmailStr, Field


class Token(BaseModel):
    """
    Schema for authentication token.
    """
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(..., description="Token type (e.g., bearer)")


class TokenData(BaseModel):
    """
    Schema for data contained in a token.
    """
    username: Optional[str] = Field(None, description="Username extracted from token")
    exp: Optional[int] = Field(None, description="Token expiration timestamp")


class UserBase(BaseModel):
    """
    Base schema for user information.
    """
    username: str = Field(..., description="Username")
    email: Optional[EmailStr] = Field(None, description="Email address")
    full_name: Optional[str] = Field(None, description="Full name")
    is_active: bool = Field(True, description="Whether the user is active")