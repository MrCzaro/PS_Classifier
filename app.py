import asyncio
from starlette.requests import Request
from starlette.datastructures import UploadFile
from starlette.responses import RedirectResponse
from passlib.hash import bcrypt_sha256 
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, Integer, String, select
from fasthtml.common import *
from monsterui.all import * 

from predictions import * 


import base64
from io import BytesIO

DATABASE_URL = "sqlite+aiosqlite:///./users.db"

engine = create_async_engine(DATABASE_URL, echo=True)
SessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()

hdrs = Theme.blue.headers()
app, rt = fast_app(hdrs=hdrs)


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
    # Pre-has with SHA256 to support passwords longer than 72 bytes
    hashed_password = bcrypt_sha256.hash(password)
    user = UserModel(login=login, password_hash=hashed_password, email=email)
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


def user_form(data=None, errors=None):
    data = data or {}
    errors = errors or {}
    return Form(
        Card(
            LabelInput("Login", name="login", type="text", value=data.get("login", ""), placeholder="Enter your login", error=errors.get("login")),
            LabelInput("Password", name="password", type="password", value=data.get("password", ""), placeholder="Enter your password", error=errors.get("password")),
            LabelInput("Repeat Password", name="repeat_password", type="password", value=data.get("repeat_password", ""), placeholder="Repeat your password", error=errors.get("repeat_password")),
            LabelInput("Email", name="email", type="email", value=data.get("email", ""), placeholder="Enter your email", error=errors.get("email")),
            header=H1("Sign Up"),
            footer = Button("Sign Up", type="submit", cls=ButtonT.primary)
        ), 
        method="post", action="/signup", id="signup-form", hx_swap="outerHTML"
    )

def login_form(data=None, errors=None):
    data = data or {}
    errors = errors or {}
    return Form(
            Card(
                LabelInput("Login", name="login", type="text", value=data.get("login", ""), placeholder="Enter your login", error = errors.get("login")),
                LabelInput("Password", name="password", type="password", value=data.get("password",""), placeholder="Enter your password", error = errors.get("password")),
                header=H1("Login"),
                footer = Button("Login", type="submit", cls=ButtonT.primary)
        ),
        method="post", action="/login", id="login-form", hx_swap="outerHTML"
        )   




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
            return user_form(data, errors)
            
        
        # Success path: create the user in the database
        await create_user(db, login=data["login"], password=data["password"], email=data["email"])

        return Card(
            P(f"Welcome, {data["login"]}! Your account has been created."), 
            header=H1("Signup Successful"),
            footer = Button("Login", hx_get="/login", hx_target="#main-content", hx_swap="innerHTML", cls=ButtonT.primary)
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
            # Pre-has the submitted password to check against the stored bcrypt hash.
            user = await get_user_by_login(db, data["login"])
            if not user or not bcrypt_sha256.verify(data["password"], user.password_hash):
                errors["login"] = "Invalid login or password."
        
        if errors:
            # re-render form with submitted values and error messaged
            return login_form(data, errors)
            
    
        # On successful login, store user info in the session
        request.session['user'] = data['login']
        # Success path 
        return Card(
            P(f"Welcome back, {data['login']}!"),
            header=H1("Login Successful"),
            footer = Button("Logout", hx_get="/logout", cls=ButtonT.primary)
        )

    


@rt("/logout")
async def sumbit_logout(request:Request):
    request.session.clear()
    return RedirectResponse("/", status_code=303)


@rt("/")
def get(request:Request):
    user = request.session.get('user')
    if user:
        content = Card(
            header=H1("Preassure Sore Classfier"),
            content=P(f"Welcome, {user}!. You are logged in."),
            footer=Button("Logout", hx_get="/logout", cls=ButtonT.secondary)
        )
    else:
        content = Card(
            header=H1("Preassure Sore Classfier"),
            content=P("Please login or sign up to continue."),
            footer=Div(
                Button("Login", hx_get="/login", hx_target="#main-content", cls=ButtonT.primary),
                cls="flex gap-2 justify-end"
            )
        )

    links = [
        Link(Button("Login",cls=ButtonT.primary), hx_get="/login", hx_target="#main-content")if not user else None,
        Link(Button("Signup", cls=ButtonT.secondary), hx_get="/signup", hx_target="#main-content") if not user else None,
        Link(Button("Logout", cls=ButtonT.secondary), hx_get="/logout") if user  else None,
    ]
    return Div(
        Header(H1(A("Pressure Sore Classifier", href="/")), cls="p-4 border-b"),
        Div(
            Aside(Nav(*[c for c in links if c is not None], cls="p-4 flex flex-col gap-2"), cls="w-64 border-r"),
            Main(content, id="main-content", cls="flex-1 p-4"),
            cls="flex flex-1"
        ),
        cls="flex flex-col h-screen"
    )
    
@rt.post("/predict_image")
async def predict_image(request: Request):
    user = request.session.get('user')
    if not user:
        return Card(P("Please log in to use the classifier."), header=H1("Access Denied"))

    form = await request.form()
    image_file: UploadFile = form.get("image_file")

    if not image_file or not image_file.filename:
        return Card(P("No image file uploaded."), header=H1("Prediction Error"))

    try:
        # Read image bytes
        contents = await image_file.read()
        # Convert bytes to PIL Image
        img = Image.open(BytesIO(contents)).convert("RGB")

        # Perform classification
        final_image_pil, message = classify_image_ps(img)

        # Convert annotated PIL Image to base64 for embedding in HTML
        buffered = BytesIO()
        final_image_pil.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return Div(
            H2("Prediction Results"),
            P(message),
            Img(src=f"data:image/jpeg;base64,{img_str}", alt="Annotated Image", cls="max-w-full h-auto"),
            cls="p-4 border rounded-lg shadow-md mt-4"
        )

    except UnidentifiedImageError:
        return Card(P("The uploaded file is not a valid image."), header=H1("Prediction Error"))
    except Exception as e:
        print(f"Prediction error: {e}")
        return Card(P(f"An error occurred during prediction: {e}"), header=H1("Prediction Error"))


@rt("/")
def get(request:Request):
    user = request.session.get('user')
    if user:
        content = Card(
            header=H1("Pressure Sore Classifier"),
            content=Div(
                P(f"Welcome, {user}!. You are logged in."),
                Form(
                    LabelInput("Upload Image", name="image_file", type="file", accept="image/*"),
                    Button("Classify Image", type="submit", cls=ButtonT.primary),
                    hx_post="/predict_image", hx_target="#prediction-output", hx_swap="innerHTML", hx_encoding="multipart/form-data",
                    cls="flex flex-col gap-4 p-4 border rounded-lg shadow-sm"
                ),
                Div(id="prediction-output", cls="mt-4") # Container for prediction results
            ),
            footer=Button("Logout", hx_get="/logout", cls=ButtonT.secondary)
        )
    else:
        content = Card(
            header=H1("Pressure Sore Classifier"),
            content=P("Please login or sign up to continue."),
            footer=Div(
                Button("Login", hx_get="/login", hx_target="#main-content", cls=ButtonT.primary),
                cls="flex gap-2 justify-end"
            )
        )

    links = [
        Link(Button("Login",cls=ButtonT.primary), hx_get="/login", hx_target="#main-content")if not user else None,
        Link(Button("Signup", cls=ButtonT.secondary), hx_get="/signup", hx_target="#main-content") if not user else None,
        Link(Button("Logout", cls=ButtonT.secondary), hx_get="/logout") if user  else None,
    ]
    return Div(
        Header(H1(A("Pressure Sore Classifier", href="/")), cls="p-4 border-b"),
        Div(
            Aside(Nav(*[c for c in links if c is not None], cls="p-4 flex flex-col gap-2"), cls="w-64 border-r"),
            Main(content, id="main-content", cls="flex-1 p-4"),
            cls="flex flex-1"
        ),
        cls="flex flex-col h-screen"
    )
    
if __name__ == "__main__":
    asyncio.run(init_db())
    serve()