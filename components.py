from dataclasses import dataclass
from datetime import datetime
from fasthtml.common import *
from monsterui.all import *


@dataclass
class LoginForm:
    email: str = ""
    password: str = ""

@dataclass
class SignupForm:
    email: str = ""
    password: str = ""
    repeat_password: str = ""


def signup_card(error_message:str|None=None, prefill_email:str=""):
    """Build the signup form card. Used for initial render and HTMX error swaps."""
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
                hx_swap="innerHTML"
            )
        ),
        CardFooter("Already have an account? ", A(B("Login"), href="#", hx_get="/login", hx_target="#content"))
    )

def login_card(error_message:str|None = None, prefill_email:str=""):
    """Build the login form card with optional error message and prefilled email."""
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
                hx_swap="innerHTML"
            )
        ),
        CardFooter("Don't have an account? ", A(B("Sign up"), href="#", hx_get="/signup", hx_target="#content"))
    )
    
def layout(req, content):
    """Compose the global page shell (nav, content slot, footer) around a given child fragment"""
    user = req.session.get("user")
    current_year = datetime.now().year

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
            f"MrCzaro © {current_year} Pressure Sore AI",
            cls="w-full p-4 bg-blue-600 text-center text-white"
        ),
        Script("document.title = 'Pressure Sore AI';"),
        Script(src="/static/preview.js"),
        cls="min-h-screen flex flex-col"
    )

def render(request, content):
    """Return only partial content for HTMX requests. Otherwise wrap in the full layout."""
    return content if request.headers.get("HX-Request") else layout(request, content)
