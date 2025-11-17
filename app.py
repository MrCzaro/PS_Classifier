import asyncio
from passlib.hash import bcrypt
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, Integer, String, select
from fasthtml.common import *

DATABASE_URL = "sqlite+aiosqlite:///./users.db"

engine = create_async_engine(DATABASE_URL, echo=True)
SessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()



# ORM model: use UserModel to avoid name clash with any dataclass
class UserModel(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    login = Column(String, unique=True, nullable=False, index=True)
    password_hash = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False)

# helper
async def init_db():
    """Creates the database table if it does not exist."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

async def get_user_by_login(db: AsyncSession, login: str):
    """Fetches a single user by their login name."""
    q = select(UserModel).where(UserModel.login == login)
    res = await db.execute(q)
    return res.scalars().first()

async def create_user(db: AsyncSession, login: str, password: str, email:str):
    """Hashes a password and creates a new user record."""
    hashed_password = bcrypt.hash(password)
    user = UserModel(login=login, password_hash=hashed_password, email=email)
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user




def user_form(data=None, errors=None):
    data = data or {}
    errors = errors or {}
    return Form(
        Fieldset(
            Label("Login", Input(name="login", type="text", value=data.get("login",""), placeholder="Enter your login")),
            Small(errors.get("login",""), style="error-text"),
            Label("Password", Input(name="password", type="password", value=data.get("password",""), placeholder="Enter your password")),
            Small(errors.get("password",""), style="error-text"),
            Label("Repeat Password", Input(name="repeat_password", type="password", value=data.get("repeat_password",""), placeholder="Repeat your password")),
            Small(errors.get("repeat_password",""), style="error-text"),
            Label("Email", Input(name="email", type="email", value=data.get("email",""), placeholder="Enter your email")),
            Small(errors.get("email",""), style="error-text"),
            Button("Sign Up", type="submit")
        ), method="post", action="/signup", id="signup-form", hx_swap="outerHTML"
    )

def login_form(data=None, errors=None):
    data = data or {}
    errors = errors or {}
    return Form(
        Fieldset(
            Label("Login", Input(name="login", type="text", value=data.get("login", ""), placeholder="Enter your login")),
            Small(errors.get("login",""), style="error-text"),
            Label("Password", Input(name="password", type="password", value=data.get("password",""), placeholder="Enter your password")),
            Small(errors.get("password",""), style="error-text"),
            Button("Login", type="submit")
        ), method="post", action="/login", id="login-form", hx_swap="outerHTML"
    )
app, rt = fast_app()

# Registration part
@rt("/signup")
async def submit_signup(request):
    async with SessionLocal() as db:
        form = await request.form()
        data = {
            "login" : form.get("login", "").strip(),
            "password" : form.get("password", "").strip(),
            "repeat_password" : form.get("repeat_password", "").strip(),
            "email" : form.get("email", "").strip(),
        }
        errors = {}

        # Valid login 
        if not data["login"]:
            errors["login"] = "Login is required."
        
        # Validate password
        if not data["password"]:
            errors["password"] = "Password is required."
        elif len(data["password"]) < 6:
            errors["password"] = "Password must be at least 6 characters long."

        # Validate repeat password
        if data["password"] != data["repeat_password"]:
            errors["repeat_password"] = "Passwords do not match."

        # Validate email
        if not data["email"]:
            errors["email"] = "Email is required."
        elif "@" not in data["email"] or "." not in data["email"]:
            errors["email"] = "Invalid email address."

        # Database validation: check if user already exists
        if not errors:
            existing_user = await get_user_by_login(db, data["login"])
            if existing_user:
                errors["login"] = "This login is already taken."
        if errors:
            # re-render form with submitted values and error messaged
            return Div(
                H1("Signup"),
                user_form(data, errors)
            )
        
        # Success path: create the user in the database
        await create_user(db, login=data["login"], password=data["password"], email=data["email"])

        return Div(
            H1("Signup Successful"),
            P(f"Welcome, {data["login"]}! Your account has been created."), 
            Button("Login", hx_get="/login", hx_target="body", hx_swap="innerHTML")
        )

@rt("/login")
async def submit_login(request):
    async with SessionLocal() as db:
        form = await request.form()
        data = {
            "login" : form.get("login", "").strip(),
            "password" : form.get("password", "").strip(),
        }
        errors = {}

        # Validate login
        if not data["login"]:
            errors["login"] = "Login is required."

        # Validate password
        if not data["password"]:
            errors["password"] = "Password is required."

        if not errors:
            user = await get_user_by_login(db, data["login"])
            if not user or not bcrypt.verify(data["password"], user.password_hash):
                errors["login"] = "Invalid login or password."
        if errors:
            # re-render form with submitted values and error messaged
            return Div(
                H1("Login"),
                login_form(data, errors)
            )
    
        # Success path 
        return Div(
            H1("Login Successful"),
            P(f"Welcome back, {data['login']}!")
        )


@rt("/")
def get():
    return Titled("Main Page", P("Demo"))

if __name__ == "__main__":
    asyncio.run(init_db())
    serve()