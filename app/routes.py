from flask import Blueprint, render_template
from app.controllers import home

main_bp = Blueprint("main", __name__)

# Show auth page first
@main_bp.route("/")
def auth():
    return render_template("auth.html")

# If user logs in, go to index page (TootHack homepage)
main_bp.add_url_rule("/home", view_func=home.home)


# Other pages
main_bp.add_url_rule("/howitworks", view_func=home.howitworks)
main_bp.add_url_rule("/upload", view_func=home.upload)
main_bp.add_url_rule("/result", view_func=home.result)
main_bp.add_url_rule("/condition", view_func=home.condition)
main_bp.add_url_rule("/predict", view_func=home.predict, methods=["POST"])
