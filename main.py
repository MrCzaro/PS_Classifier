
import sqlite3, base64
from io import BytesIO
from pathlib import Path
from PIL import Image
from urllib.parse import quote_plus, unquote_plus

from fasthtml.common import *
from monsterui.all import *
#from starlette.responses import RedirectResponse

from components  import *
from passwords_helper import hash_password, verify_password
from ps_classifier import classify_image_ps, pressure_examples, no_pressure_examples





### Data & Config
EXAMPLES = pressure_examples + no_pressure_examples
DB_PATH = "users.db"
login_redir = RedirectResponse("/login", status_code=303)

### Database helpers
def get_db():
    """Opens a SQLite connection to the users database with row access by column name."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Create the users table if needed and initialize the database schema."""

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

### Initialize DB on startup
init_db()


### Auth beforeware
def require_auth(request, session):
    """Beforeware guard that redirects to /login when no user is in session."""
    user = session.get("user")
    if not user:
        return login_redir
    request.scope["user"] = user

before = Beforeware(
    require_auth,
    skip=[r"/favicon\.ico", r"/static/.*", "/login", "/signup"]
)

### App setup
hdrs = Theme.blue.headers()
app, rt = fast_app(hdrs=hdrs, static_path="static", before=before)


### Routers ###
@rt("/favicon.ico")
def favicon(req):
    # redirect the browser to the static file
    return Redirect("/static/favicon.ico") 

# --- HOME ROUTE ---
@rt("/")
async def index(req, sess):
    """Render the authenticated home view with examples, upload area, and prediction panel."""
    example_cards = []
    for img_path in EXAMPLES:
        name = Path(img_path).name
        url_path = quote_plus(img_path)

        example_cards.append(
            Div(
                Img(src=f"/static/{name}",cls="w-44 h-44 object-cover rounded-lg shadow"),
                A(
                    Button("Classify",cls=ButtonT.primary + " mt-2 w-full rounded-md py-3 px-5 md:py-3 md:px-6"),
                    hx_get=f"/classify?img_path={url_path}",
                    hx_target="#prediction-output",
                    hx_swap="outerHTML"
                ),
                cls="flex flex-col items-center p-3 border rounded-xl shadow-sm bg-white flex-shrink-0 w-48 snap-center"
            )
        )

    # Make a single-row scroller for the examples (no wrapping on large screens)
    example_grid = Div(
        Div(*example_cards, cls="flex gap-4 items-start overflow-x-auto snap-x snap-mandatory py-2 px-2"),
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

    return render(req, content)

# --- CLASSIFICATION ROUTE ---
@rt("/classify")
async def classify(req, image_path: str):
    """Load a selected example image, run the classifier, and return the prediction fragment."""

    if not img_path:
        return Div(P("No image specified. Click an example or upload one.", cls="text-red-600"),
                   id="prediction-output")

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

@rt("/upload-classify")
async def upload_classify(file: UploadFile):
    """Accept an uploaded image, run classification, and return the result fragment."""
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


@rt("/login")
def login(req, sess, form: LoginForm | None = None):
    """Serve the login form on Get. Validate credentials and start the session on POST."""
    if req.method == "GET" or form is None:
        return render(req, login_card())
    

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
            
    sess["user"] = email
    return Redirect("/")

# --- SIGNUP ROUTES ---
@rt("/signup")
def signup(req, sess, form: SignupForm | None = None):
    """Serve the signup form on GET. Create a user record and log in on POST."""
    if req.method == "GET" or form is None:
        return render(req, signup_card())

    email = form.get("email", "").strip().lower()
    password = form.get("password", "").strip()
    repeat_password = form.get("repeat_password", "").strip()

    if password != repeat_password:
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
    req.session["user"] = email
    return Redirect("/")

# --- LOGOUT ROUTE ---
@rt("/logout")
def logout(sess):
    """Clear the session and redirect to the login page."""
    sess.clear()
    return Redirect("/")

serve()