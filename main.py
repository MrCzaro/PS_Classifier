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
from urllib.parse import quote_plus, unquote_plus

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
                Div(Button("Sign Up", cls=ButtonT.primary + " rounded-lg py-2 px-4 md:py-3 md:px-5 text-sm md:text-base", type="submit"), cls="mt-4"),
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
                Div(Button("Login", cls=ButtonT.primary + " rounded-lg py-2 px-4 md:py-3 md:px-5 text-sm md:text-base", type="submit"), cls="mt-4"),
                action="/login",
                method="post",
                hx_post="/login",
                hx_target="#content",
                hx_swap="outerHTML"
            )
        ),
        CardFooter("Don't have an account? ", A(B("Sign up"), href="#", hx_get="/signup", hx_target="#content"))
    )
    



@app.get("/favicon.ico")
def favicon(request):
    # redirect the browser to the static file
    return Redirect("/static/favicon.ico") 

def layout(request, content):
    """
    Layout wrapper: Top navbar + page content + sticky footer.
    Centers content and prevents footer overlap by giving the content area flex:1.
    """
    user = request.session.get("user")

    logo = A("Pressure Sore AI", href="/", cls="text-xl font-bold text-white tracking-tight")

    links = [
        A(Button("Login", cls=ButtonT.primary + " rounded-lg py-2 px-4 md:py-3 md:px-5 text-sm md:text-base"), hx_get="/login", hx_target="#content") if not user else None,
        A(Button("Signup", cls=ButtonT.secondary + " rounded-lg py-2 px-4 md:py-3 md:px-5 text-sm md:text-base"), hx_get="/signup", hx_target="#content") if not user else None,
        A(Button("Logout", cls=ButtonT.secondary + " rounded-lg py-2 px-4 md:py-3 md:px-5 text-sm md:text-base"), hx_get="/logout") if user else None,
    ]
    links = [c for c in links if c is not None]

    nav_bar = Nav(
        Div(logo, cls="flex items-center"),
        Div(*links, cls="flex gap-2 items-center"),
        cls="flex justify-between items-center bg-blue-600 px-4 py-2"
    )


    return Div(
        Header(nav_bar),
        Div(
            Container(content, id="content", clsx="mt-10 max-w-6xl w-full"), 
            cls="flex-1 w-full"
        ),
        Footer(
            "MrCzaro © 2025 Pressure Sore AI",
            cls="w-full p-4 bg-blue-600 text-center text-white"
        ),
        Script(src="/static/preview.js"),
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
    # Example Cards - build one card per example
    example_cards = []
    for img_path in EXAMPLES:
        name = Path(img_path).name
        url_path = quote_plus(img_path)

        example_cards.append(
            Div(
                Img(src=f"/static/{name}",
                    cls="w-44 h-44 object-cover rounded-lg shadow"),
                A(
                    Button(
                        "Classify",
                        cls=ButtonT.primary + " mt-2 w-full rounded-md py-3 px-5 md:py-3 md:px-6"
                    ),
                    hx_get=f"/classify?img_path={url_path}",
                    hx_target="#prediction-output",
                    hx_swap="outerHTML"
                ),
                cls="flex flex-col items-center p-3 border rounded-xl shadow-sm bg-white flex-shrink-0 w-48 snap-center"
            )
        )

    # Make a single-row scroller for the examples (no wrapping on large screens)
    example_grid = Div(
        Div(
            *example_cards,
            cls="flex gap-4 items-start overflow-x-auto snap-x snap-mandatory py-2 px-2",
        ),
        cls="w-full"
    )

    # Upload Area and Prediction Area
    upload_area = Card(
        CardHeader(H3("Upload Image")),
        CardBody(
            Div(
                # Drop Zone
                Div(
                    P("Drag & drop an image here", cls="text-gray-600 text-sm"),
                    P("or click to browse", cls="text-blue-600 text-sm font-semibold"),
                    Input(
                        type="file", name="file", id="userfile",
                        accept="image/*",
                        cls="hidden",
                        onchange="previewFile(event)"
                    ),
                    id="drop-zone",
                    cls="border-2 border-dashed border-blue-400 rounded-xl p-6 text-center cursor-pointer bg-white hover:bg-blue-50 transition"
                ),
                Div(
                    Img(id="preview-img", cls="hidden mx-auto mt-4 max-h-64 rounded-lg shadow-md"),
                    # metadata / filename spot (JS can populate this)
                    Div(id="preview-meta", cls="text-sm text-gray-600 mt-2 text-center"),
                    Button(
                        "Classify Image",
                        cls=ButtonT.primary + " hidden mt-4 w-full rounded-md py-3 px-5",
                        id="upload-btn",
                        hx_post="/upload-classify",
                        hx_target="#prediction-output",
                        hx_swap="outerHTML",
                        hx_encoding="multipart/form-data",
                        hx_include="#userfile"
                    ),
                    cls="mt-2"
                ),

                cls="w-full"
            )
        ),
        cls="w-full"
    )

    prediction_area = Card(
        CardHeader(H3("Prediction Result")),
        CardBody(
            Div("No image classified yet.", id="prediction-output", cls="text-gray-600 text-left"),
        ),
        cls="w-full"
    )

    two_column_layout = Div(
        Div(upload_area, cls="col-span-12 md:col-span-4"),
        Div(prediction_area, cls="col-span-12 md:col-span-8"),
        cls="grid grid-cols-12 gap-6 mt-10 items-start"
    )

    content = Div(
        H2("Pressure Sore Classifier", cls="text-3xl font-bold mb-6 text-center"),
        H3("Example Images", cls="text-xl font-semibold mb-3"),
        example_grid,
        two_column_layout,
        cls="p-6"
    )

    return render(request, content)


# --- CLASSIFICATION ROUTE ---
@app.get("/classify")
@login_required
async def classify(request):

    img_path =  request.query_params.get("img_path")

    # Debug logging (temporary)
    print("DEBUG /classify (GET) img_path:", img_path)

    if not img_path:
        return Div(P("No image specified. Click an example or upload one.", cls="text-red-600"),
                   id="prediction-output")

    # URL-decoding if you used quote_plus in hx_get
    img_path = unquote_plus(img_path)
    p = Path(img_path)
    if not p.exists():
        candidate = Path("static") / p.name
        if candidate.exists():
            img_path = str(candidate)
        else:
            return Div(P(f"Image not found: {p.name}", cls="text-red-600"), id="prediction-output")

    final_image, message = classify_image_ps(img_path)

    if final_image is None:
        return Div(
            P("Classification failed.", cls="text-red-600 font-semibold"),
            P(message, cls="text-gray-700"),
            id="prediction-output"
        )

    buffer = BytesIO()
    final_image.save(buffer, format="JPEG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    img_src = f"data:image/jpeg;base64,{encoded}"

    content = Div(
        Img(src=img_src, cls="block max-w-full md:max-w-md rounded-lg shadow-lg"),
        P(message, cls="mt-3 font-semibold text-lg"),
        id="prediction-output"
    )

    return content

@app.post("/upload-classify")
@login_required
async def upload_classify(request):
    form = await request.form()

    file = form.get("file")
    if not file:
        return Div("No file uploaded.", cls="text-red-600", id="prediction-output")
    
    # Read into PIL
    img_bytes = await file.read()
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    tmp_path = "static/_tmp_upload.jpg"
    img.save(tmp_path)

    final_image, message = classify_image_ps(tmp_path)

    if final_image is None:
        return Div(
            P("Classification failed.", cls="text-read-600 font-semibold"),
            P(message, cls="text-gray-700"),
            id="prediction-output"
        )

    # Convert to Base64 for display
    buf = BytesIO()
    final_image.save(buf, format="JPEG")
    encoded = base64.b64encode(buf.getvalue()).decode()

    return Div(
        Img(src=f"data:image/jpeg;base64,{encoded}", cls="block max-w-full md:max-w-md rounded-lg shadow-lg"),
        P(message, cls="mt-3 font-semibold text-lg"),
        id="prediction-output"
    )

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

    # if request came from HTMX, redirect to client-side
    if request.headers.get("HX-Request"):
        return Response(status_code=200, headers={"HX-Redirect": "/"})
    
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