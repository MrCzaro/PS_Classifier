from fasthtml.common import *
from monsterui.all import *

### Application setup
hdrs = Theme.blue.headers()
app = FastHTML(hdrs=hdrs)

### Helper functions
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
        A(Button("Signup", cls=ButtonT.secondary), hx_get="/signup", hx_target="#content") if not user else None,
        A(Button("Logout", cls=ButtonT.secondary), hx_get="/logout") if user else None,
    ]
    links = [c for c in links if c is not None]

    # Navbar container: flex, justify-between, bg-blue
    nav_bar = Nav(
        Div(logo, cls="flex items-center"),         # left side
        Div(*links, cls="flex gap-2 items-center"), # right side
        cls="flex justify-between items-center bg-blue-600 px-4 py-2"
    )

    return Div(
        Header(nav_bar),
        Div(
            Container(content, id="content", cls="mt-10 max-w-lg"),
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

### Users
users = {
    "testuser@test.com": "password123",
    "test_user@test.com": "test_user123"
}

### Routers

# --- HOME ROUTE ---
@app.get("/")
def index(request):
    user = request.session.get("user")
    if user:
        content = Card(
            CardHeader(B(f"Hello, {user}!")),
            CardBody("You are now logged in and can access the classification tools."),
        )
    else:
        content = Card(
            CardHeader(B("Welcome")),
            CardBody("Please login or signup to access the application.")
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
    

    if email == "" or password == "":
        return login_card("All fields are required.", prefill_email=email)
    
    if email not in users or users[email] != password:
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

    if email in users:
        return signup_card("User already exists. Please login.", prefill_email=email)
    
    # 3. Success: Add to DB and Log them in
    users[email] = password
    
    request.session["user"] = email
    
    return Redirect("/")

# --- LOGOUT ROUTE ---

@app.get("/logout")
def logout(request):
    request.session.clear()
    return Redirect("/")

if __name__ == "__main__":
    serve()