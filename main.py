from fasthtml.common import *
from monsterui.all import *

### Application setup
app, rt = fast_app()

### Helper functions
def signup_card(error_message=None, prefill_email=""):
    """Identical structure to login, but for registration."""
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
                Div(Button("Sign Up", cls=ButtonT.primary, type="submit"), cls="mt-4"),
                hx_post="/signup",
                hx_target="#content",
                hx_swap="outerHTML"
            )
        ),
        CardFooter("Already have an account? ", A("Login", href="#", hx_get="/login", hx_target="#content"))
    )

def login_card(error_message=None, prefill_email=""):
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
                hx_post="/login",
                hx_target="#content",
                hx_swap="outerHTML"
            )
        ),
        CardFooter("Don't have an account? ", A("Sign up", href="#", hx_get="/signup", hx_target="#content"))
    )

def nav(request):
    """Returns a session-aware navigation bar."""
    user = request.session.get("user")
    
    
    brand = A("Pressure Sore AI", href="/", cls="text-xl font-bold text-primary tracking-tight")
    
    if user:
        right_side = A(Button("Logout", cls=ButtonT.secondary), href="/logout")
    else:
        login_btn = A(Button("Login", cls=ButtonT.ghost), 
                      hx_get="/login", 
                      hx_target="#content")
        
        signup_btn = A(Button("Signup", cls=ButtonT.primary), 
                       hx_get="/signup", 
                       hx_target="#content")
        
        right_side = Div(login_btn, signup_btn, cls="space-x-2")
    
    return NavBar(brand, right_side)

def layout(request, content):
    return Html(
        Head(Title("Pressure Sore App")),
        Body(
            nav(request),
            Container(content, id="content", cls="mt-10 max-w-lg"),
            Footer("MrCzaro © 2025", cls="p-4 text-center text-gray-500")
        )
    )

def render_page(request, content):
    """Decides whether to return full layout or just the partial content."""
    if request.headers.get('hx-request'):
        return content
    return layout(request, content)


### Users
users = {
    "testuser@test.com": "password123",
    "test_user@test.com": "test_user123"
}

### Routers
@rt("/")
def home(request):
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
    return render_page(request, content)

# --- LOGIN ROUTES ---

@rt("/login", methods=["GET"])
def get_login(request):
    return render_page(request, login_card())

@rt("/login", methods=["POST"])
async def post_login(request):
    form = await request.form()
    email = form.get("email", "").strip().lower()
    password = form.get("password", "").strip()

    if email not in users or users[email] != password:
        return login_card("Invalid email or password.", prefill_email=email)
    
    request.session["user"] = email
    return Response(headers={"HX-Location": "/"})

# --- SIGNUP ROUTES ---

@rt("/signup", methods=["GET"])
def get_signup(request):
    return render_page(request, signup_card())

@rt("/signup", methods=["POST"])
async def post_signup(request):
    form = await request.form()
    email = form.get("email", "").strip().lower()
    password = form.get("password", "").strip()

    # 1. Validation: Check if empty
    if not email or not password:
        return signup_card("All fields are required.", prefill_email=email)

    # 2. Validation: Check if user exists
    if email in users:
        return signup_card("User already exists. Please login.", prefill_email=email)
    
    # 3. Success: Add to DB and Log them in
    users[email] = password
    request.session["user"] = email
    
    # 4. Redirect to home (Updating the navbar)
    return Response(headers={"HX-Location": "/"})

@rt("/logout")
def logout(request):
    request.session.clear()
    return Redirect("/")

if __name__ == "__main__":
    serve()