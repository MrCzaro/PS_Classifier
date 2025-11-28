import bcrypt

def hash_password(plain_password: str) -> str:
    """
    Return a bcrypt-hashed version of the plaintext password."""
    hashed = bcrypt.hashpw(plain_password.encode("utf-8"), bcrypt.gensalt())
    if isinstance(hashed, bytes):
        return hashed.decode("utf-8")
    return hashed

def verify_password(plain_password:str, hashed_password:str) -> bool:
    """
    Verify a plaintext password against the hashed version.
    """
    return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode())

