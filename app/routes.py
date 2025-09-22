from flask import Blueprint
from app.controllers import home

main_bp = Blueprint("main", __name__)

# Home
main_bp.add_url_rule("/", view_func=home.home)

# Other pages
main_bp.add_url_rule("/howitworks", view_func=home.howitworks)
main_bp.add_url_rule("/upload", view_func=home.upload)
main_bp.add_url_rule("/result", view_func=home.result)
main_bp.add_url_rule("/condition", view_func=home.condition)
main_bp.add_url_rule("/predict", view_func=home.predict, methods=["POST"])