from functools import wraps
from inspect import iscoroutinefunction

from monsterui.all import *
from fasthtml.common import *
from starlette.staticfiles import StaticFiles

import sqlite3, base64
from passwords_helper import hash_password, verify_password


from ps_classifier import classify_image_ps, pressure_examples, no_pressure_examples
from io import BytesIO
from pathlib import Path
from PIL import Image
EXAMPLES = pressure_examples + no_pressure_examples
### Initialize or connect to the database

DB_PATH = "users.db"

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    db = get_db()
    db.execute("""
               CREATE TABLE IF NOT EXISTS users(
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               email TEXT UNIQUE NOT NULL,
               password_hash TEXT NOT NULL,
               created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
               )
               """)
    db.commit()
    db.close()

# Initialize DB on startup
init_db()


### Application setup
hdrs = Theme.blue.headers()
app = FastHTML(hdrs=hdrs, static_dir="static")
app.mount("/static", StaticFiles(directory="static"), name="static")
db = get_db()


### Helper functions
def login_required(route_func):
    """
    Restrict access to authenticated users in FastHTML routes.
    Works for both sync and async route handlers.
    Redirects to `/login` if no `user` key exists in session.
    """
    @wraps(route_func)
    async def wrapper(request, *args, **kwargs):
        if not request.session.get("user"):
            return Redirect("/login")
        # If the orginal route is async, await it 
        if iscoroutinefunction(route_func):
            return await route_func(request, *args, **kwargs)
        
        # If normal sync function
        
        return route_func(request, *args, **kwargs)
    return wrapper

def signup_card(error_message:str|None=None, prefill_email:str=""):
    """Signup card partial (returned by HTMX on POST errors)."""
    return Card(
        CardHeader(H3("Create Account")),
        CardBody(
            *([P(error_message, cls="text-red-600 font-semibold")] if error_message else []),
            Form(
                LabelInput("Email", name="email", id="email",
                           placeholder="user@example.com",
                           value=prefill_email),
                LabelInput("Password", name="password", id="password",
                           type="password", placeholder="Choose a password"),
                LabelInput("Repeat Password", name="repeat_password", id="repeat_password",
                           type="password", placeholder="Repeat password"),
                Div(Button("Sign Up", cls=ButtonT.primary, type="submit"), cls="mt-4"),
                action="/signup",
                method="post",
                hx_post="/signup",
                hx_target="#content",
                hx_swap="outerHTML"
            )
        ),
        CardFooter("Already have an account? ", A(B("Login"), href="#", hx_get="/login", hx_target="#content"))
    )

def login_card(error_message:str|None = None, prefill_email:str=""):
    """
    Returns a login form card.
    error_message: optional string to display errors
    prefill_email: email to pre-fill the form
    """
    return Card(
        CardHeader(H3("Login")),
        CardBody(
            *([P(error_message, cls="bg-red-100 border border-red-300 text-red-700 p-2 rounded")] if error_message else []),
            Form(
                LabelInput("Email", name="email", id="email",
                           placeholder="user@example.com",
                           value=prefill_email),
                LabelInput("Password", name="password", id="password",
                           type="password", placeholder="Password"),
                Div(Button("Login", cls=ButtonT.primary, type="submit"), cls="mt-4"),
                action="/login",
                method="post",
                hx_post="/login",
                hx_target="#content",
                hx_swap="outerHTML"
            )
        ),
        CardFooter("Don't have an account? ", A(B("Sign up"), href="#", hx_get="/signup", hx_target="#content"))
    )
    



def layout(request, content):
    """
    Centered container layout with styled navbar.
    """
    user = request.session.get("user")

    # Logo
    logo = A("Pressure Sore AI", href="/", cls="text-xl font-bold text-white tracking-tight")

    # Links / buttons
    links = [
        A(Button("Login", cls=ButtonT.primary), hx_get="/login", hx_target="#content") if not user else None,
        A(Button("Signup", cls=ButtonT.secondary,), hx_get="/signup", hx_target="#content") if not user else None,
        A(Button("Logout", cls=ButtonT.secondary), hx_get="/logout") if user else None,
    ]
    links = [c for c in links if c is not None]

    # Navbar container: flex, justify-between, bg-blu
    nav_bar = Nav(
        Div(logo, cls="flex items-center"),         # left side
        Div(*links, cls="flex gap-2 items-center"), # right side
        cls="flex justify-between items-center bg-blue-600 px-4 py-2"
    )

    return Div(
        Header(nav_bar),
        Div(
            Container(content, id="content", clsx="mt-10 max-w-lg"),
            Footer(
                    "MrCzaro © 2025 Pressure Sore AI",
                    cls="fixed bottom-0 left-0 w-full p-4 bg-blue-600 backdrop-blur text-center text-white"
                ),
            cls="flex flex-col min-h-screen"
        ),
        cls="min-h-screen flex flex-col"
    )

def render(request, content):
    """
    If the request comes from HTMX (a navbar click), return only the content.
    IF it is a direct browser visit, return the full layout.
    """
    if request.headers.get("HX-Request"):
        return content
    else:
        return layout(request, content)


### Routers

# --- HOME ROUTE ---
@app.get("/")
@login_required
async def index(request):
    cards = []
    for img_path in EXAMPLES:
        cards.append(
            Div(
                Img(src=f"/static/{Path(img_path).name}", cls="w-40 rounded shadow"),
                Button("Classify", cls=ButtonT.primary,
                       hx_post="/classify",
                       hx_target="#result",
                       hx_vals={"img_path": img_path}
            ),
            cls="flex flex-col items-center m-3"
        )
    )
    

    content = Div(
        H2("Pressure Sore Examples", cls="text-2xl font-bold mb-4"),
        Div(*cards, cls="flex flex-wrap gap-4 justify-center"),
        Div(id="result", cls="mt-10"),
        cls="p-6"
    )

    return render(request, content)

@app.post("/classify")
@login_required
async def classify(request):
    data = await request.form()
    img_path = data.get("img_path")

    final_image, message = classify_image_ps(img_path)

    # Convert PIL to base64 to display in HTML
    buffer = BytesIO()
    final_image.save(buffer, format="JPEG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    img_src = f"data:image/jpeg;base64,{encoded}"

    content = Div(
        Img(src=img_src, cls="max-w-md rounded-lg shadow-lg"),
        P(message, cls="mt-3 font-semibold text-lg"),
        id="result"
    )

    return render(request, content)

# --- LOGIN ROUTES ---

@app.get("/login")
def get_login(request):
    return render(request, login_card())

@app.post("/login")
async def post_login(request):
    form = await request.form()
    email = form.get("email", "").strip().lower()
    password = form.get("password", "").strip()
    
    if not email or not password:
        return login_card("All fields are required.", prefill_email=email)

    # Fetch user from DB
    db = get_db()
    cur = db.execute("SELECT * FROM users WHERE email = ?", (email,))
    user = cur.fetchone()
    db.close()
    if not user or not verify_password(password, user["password_hash"]):
        return login_card("Invalid email or password.", prefill_email=email)
            
    request.session["user"] = email
    return Redirect("/")


# --- SIGNUP ROUTES ---

@app.get("/signup")
def signup(request):
    return render(request, signup_card())

@app.post("/signup")
async def post_signup(request):
    form = await request.form()
    email = form.get("email", "").strip().lower()
    password = form.get("password", "").strip()
    repreat_password = form.get("repeat_password", "").strip()

    if password != repreat_password:
        return signup_card("Passwords do not match.", prefill_email=email)
    
    if not email or not password:
        return signup_card("All fields are required.", prefill_email=email)

    db = get_db()
    # Check if user exists
    cur = db.execute("SELECT id FROM users WHERE email = ?", (email,))
    if cur.fetchone():
        db.close()
        return signup_card("User already exists. Please login.", prefill_email=email)
    
    
    
    # Save user into DB
    hashed = hash_password(password)
    db.execute("INSERT INTO users (email, password_hash) VALUES (?, ?)", (email, hashed))
    db.commit()
    db.close()
    
    request.session["user"] = email
    
    return Redirect("/")

# --- LOGOUT ROUTE ---

@app.get("/logout")
def logout(request):
    request.session.clear()
    return Redirect("/")

#if __name__ == "__main__":
serve()