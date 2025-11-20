from fasthtml.common import *
from monsterui.all import *

### Application setup
hdrs = Theme.blue.headers()
app = fast_app(hdrs=hdrs)

### Helper functions
def signup_card(error_message:str|None=None, prefill_email:str=""):
    """Signup card partial (returned by HTMX on POST errors)."""
    return Card(
        CardHeader("Create Account"),
        CardBody(
            *([P(error_message, cls=TextT.red)] if error_message else []),
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
        CardFooter("Already have an account? ", A("Login", href="#", hx_get="/login", hx_target="#content"))
    )

def login_card(error_message:str|None = None, prefill_email:str=""):
    """
    Returns a login form card.
    error_message: optional string to display errors
    prefill_email: email to pre-fill the form
    """
    return Card(
        CardHeader("Login"),
        CardBody(
            *([P(error_message, cls=TextT.red)] if error_message else []),
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
        CardFooter("Don't have an account? ", A("Sign up", href="#", hx_get="/signup", hx_target="#content"))
    )
    

def nav(request):
    """
    Session-aware navbar. Buttons that load partials use hx_get into #content.
    Logout uses a norma href (full reload) so Nav re-renders immediately.
    """
    user = request.session.get("user")

    brand = A("Pressure Sore AI", href="/", cls="text-xl font-bold text-primary tracking-tight")

    if user:
        right_side = A(
            Button("Logout", cls=ButtonT.secondary), href="/logout")
    else:
        login_btn = A(Button("Login", cls=ButtonT.ghost), hx_get="/login", hx_target="#content")

        signup_btn = A(Button("Signup", cls=ButtonT.primary), hx_get="/signup", hx_target="#content")

        right_side = Div(login_btn, signup_btn, cls="space-x-2")

    return NavBar(brand, right_side)


def layout(request, content):
    """
    Centered container layout.
    """
    return Html(
        Head(Title("Pressure Sore App")),
        Body(
            nav(request),
            Container(content, id="content", cls="mt-10 max-w-lg"),
            Footer("MrCzaro © 2025", cls="p-4 text-center text-gray-500")
        )
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
def get(request):
    user = request.session.get("user")
    if user:
        content = Card(
            CardHeader(f"Hello, {user}!"),
            CardBody("You are now logged in and can access the classification tools."),
        )
    else:
        content = Card(
            CardHeader("Welcome"),
            CardBody("Please login or signup to access the application.")
        )
    return layout(request, content)

# --- LOGIN ROUTES ---

@app.get("/login")
def get_login(request):
    return layout(request, login_card())

@app.post("/login")
async def post_login(request):
    form = await request.form()
    email = form.get("email", "").strip().lower()
    password = form.get("password", "").strip()
    

    if email == "" or password == "":
        return login_card("All fields are required.", prefill_email=email)
    
    if email not in users or users[email] != password:
        return login_card(request,login_card("Invalid email or password.", prefill_email=email))
    
    request.session["user"] = email
    return HtmxResponseHeaders(location="/")

# --- SIGNUP ROUTES ---

@app.get("/signup")
def signup(request):
    return layout(request, signup_card())

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
    
    return HtmxResponseHeaders(location="/")

# --- LOGOUT ROUTE ---

@app.get("/logout")
def logout(request):
    request.session.clear()
    return Redirect("/")

if __name__ == "__main__":
    serve()